"""DeepGEMM-backed MHC pre-GEMM resolved wrapper.

Lives in its own module rather than `srt/layers/mhc.py` so that the
deepgemm path stays free of the TileLang import at module top
(`mhc.py:5-6` imports `tilelang` and `tilelang.language as T`).

Two layers exposed:

- `deepgemm_hc_pre_gemm_splitk(x_flat, fn, num_splits)`: thin wrapper
  around `deep_gemm.tf32_hc_prenorm_gemm` with explicit split-K.
  Returns 3D `gemm_out [S, M, N]` and 2D `sqrsum [S, M]` partials —
  the same layout `mhc_pre_big_fuse_triton` consumes when called
  with `n_splits=S>1`.
- `mhc_pre_deepgemm(residual, fn, ...) -> (post, comb, layer_input)`:
  full resolved callable; binds to `self._mhc_pre_fn` in
  `DeepseekV4DecoderLayer.__init__`. Body calls
  `deepgemm_hc_pre_gemm_splitk` for the GEMM step and
  `mhc_pre_big_fuse_triton` for the sinkhorn + per-head normalization
  + output combination + split-K reduction step.

## Why num_splits matters

The sm_120 `tf32_hc_prenorm_gemm` kernel sets `BLOCK_M=128` and grid =
`ceil_div(M, 128) * num_splits`. With `num_splits=None` (i.e. 1) and
`M<=128` (V4-Flash decode), grid_size==1 — a single CTA serializes
all 256 K-blocks (K=16384, BLOCK_K=64). That leaves 83 of 84 SMs idle
on RTX PRO 6000 and produces a flat ~143 µs floor regardless of M.

With `num_splits=32` the same shape gets 32 CTAs of work and drops
to ~7 µs. With `num_splits=128`, ~4 µs. End-to-end (GEMM + big_fuse
reduction) at decode shapes the deepgemm split-K path is ~33 µs vs
the Triton path's ~67 µs — a 2× win.

See `_test_mhc_pre_deepgemm.py` and `/tmp/bench_mhc_pre_full.py` for
the data backing the `num_splits=32` default below.
"""

from __future__ import annotations

from typing import Tuple

import torch


# Default split-K factor for V4-Flash decode shapes. S=32 gives near-best
# end-to-end at every M ∈ {1..2048}; S=128 wins at small M but loses at
# M=2048 (more partials than the big_fuse reduction can amortize).
_DEFAULT_NUM_SPLITS_DECODE = 32


def deepgemm_hc_pre_gemm_splitk(
    x_flat: torch.Tensor,
    fn: torch.Tensor,
    num_splits: int = _DEFAULT_NUM_SPLITS_DECODE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BF16 × FP32 → FP32 GEMM with per-row sum-of-squares, split-K.

    Args:
        x_flat: [M, K] BF16, K-major (M = num_tokens, K = hc_hidden_size).
        fn:     [N, K] FP32, K-major (N = hc_mult3).
        num_splits: split-K factor. Public API requires 3D `d`/2D
            `sqr_sum` whenever this is provided. Each split holds the
            partial GEMM output for K-segment `[s*K/S .. (s+1)*K/S)`
            and the partial sum-of-squares for the same K-segment.
            `mhc_pre_big_fuse_triton` reduces all S splits.

    Returns:
        gemm_out: [S, M, N] FP32 partials, contiguous.
        sqrsum:   [S, M]    FP32 partials, contiguous.
    """
    assert x_flat.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert x_flat.dim() == 2 and fn.dim() == 2
    assert x_flat.shape[1] == fn.shape[1], (
        f"K mismatch: x_flat.shape[1]={x_flat.shape[1]} vs fn.shape[1]={fn.shape[1]}"
    )
    assert num_splits >= 1

    import deep_gemm

    M, _ = x_flat.shape
    N, _ = fn.shape

    gemm_out = torch.empty(
        num_splits, M, N, dtype=torch.float32, device=x_flat.device,
    )
    sqrsum = torch.empty(
        num_splits, M, dtype=torch.float32, device=x_flat.device,
    )

    deep_gemm.tf32_hc_prenorm_gemm(
        x_flat, fn, gemm_out, sqrsum, num_splits=num_splits,
    )
    return gemm_out, sqrsum


def mhc_pre_deepgemm(
    residual: torch.Tensor,        # [..., hc_mult, hidden] bf16
    fn: torch.Tensor,              # [hc_mult3, hc_mult * hidden] fp32
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    n_splits_pre: int = _DEFAULT_NUM_SPLITS_DECODE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in deepgemm-backed replacement for `mhc_pre_triton`.

    GEMM via `deep_gemm.tf32_hc_prenorm_gemm` with `num_splits=n_splits_pre`;
    sinkhorn + per-head normalization + split-K reduction + output
    combination via `mhc_pre_big_fuse_triton` (deepgemm has no big_fuse
    equivalent).

    The `n_splits` parameter is the OUTER reduction factor consumed by
    the original `mhc_pre_triton` interface (1 by default; the Triton
    split-K wrapper produced 1 split internally and the big_fuse
    interface re-uses `n_splits` to mean "outer split factor"). For
    deepgemm we use `n_splits_pre` as the GEMM-internal split factor;
    big_fuse then reduces those `n_splits_pre` partials.

    Returns (post_mix [..., hc_mult, 1], comb_mix [..., hc_mult,
    hc_mult], layer_input [..., hidden]).
    """
    del n_splits  # The legacy "outer" split factor; we use n_splits_pre.

    from sglang.jit_kernel.mhc_triton import mhc_pre_big_fuse_triton

    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape == (hc_mult3, hc_hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device,
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device,
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device,
    )

    if num_tokens == 0:
        post_mix = post_mix.view(*outer_shape, hc_mult, 1)
        comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
        layer_input = layer_input.view(*outer_shape, hidden_size)
        return post_mix, comb_mix, layer_input

    # Stage 1: split-K GEMM + per-row sum-of-squares via deepgemm.
    # 3D output [S, M, hc_mult3]; 2D sqrsum [S, M].
    x_flat = residual_flat.view(num_tokens, hc_hidden_size)
    gemm_out_3d, gemm_sqr_2d = deepgemm_hc_pre_gemm_splitk(
        x_flat, fn, num_splits=n_splits_pre,
    )

    # Stage 2: big_fuse reduces the S K-partials, applies sinkhorn,
    # per-head normalization, and writes (post, comb, layer_input).
    mhc_pre_big_fuse_triton(
        gemm_out_3d, gemm_sqr_2d, hc_scale, hc_base, residual_flat,
        post_mix, comb_mix, layer_input,
        hidden_size=hidden_size,
        rms_eps=rms_eps, hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        n_splits=n_splits_pre, hc_mult=hc_mult,
    )

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)
    return post_mix, comb_mix, layer_input
