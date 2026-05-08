"""DeepGEMM-backed MHC pre-GEMM resolved wrapper.

Lives in its own module rather than `srt/layers/mhc.py` so that the
deepgemm path stays free of the TileLang import at module top
(`mhc.py:5-6` imports `tilelang` and `tilelang.language as T`).

Two layers exposed:

- `deepgemm_hc_pre_gemm(x_flat, fn) -> (gemm_out, sqrsum)`: thin
  wrapper around `deep_gemm.tf32_hc_prenorm_gemm`. Same `(out,
  sqrsum)` contract the existing `mhc_pre_big_fuse_triton` consumes.
- `mhc_pre_deepgemm(residual, fn, ...) -> (post, comb, layer_input)`:
  full resolved callable; binds to `self._mhc_pre_fn` in
  `DeepseekV4DecoderLayer.__init__`. Body calls
  `deepgemm_hc_pre_gemm` for the GEMM step and
  `mhc_pre_big_fuse_triton` for the sinkhorn + per-head normalization
  + output combination step (deepgemm has no big_fuse equivalent).
"""

from __future__ import annotations

from typing import Tuple

import torch


def deepgemm_hc_pre_gemm(
    x_flat: torch.Tensor,
    fn: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """BF16 × FP32 → FP32 GEMM with per-row sum-of-squares.

    `a` is BF16 [m, k] K-major; `b` is FP32 [n, k] K-major. Output `d`
    is FP32 [m, n] N-major; `sqr_sum` is FP32 [m] holding the per-row
    sum-of-squares of `a` (the input rows fed into the GEMM).

    Args:
        x_flat: [num_tokens, hc_hidden_size] BF16, K-major.
        fn:     [hc_mult3, hc_hidden_size]   FP32, K-major.

    Returns:
        gemm_out: [num_tokens, hc_mult3] FP32, N-major contiguous.
        sqrsum:   [num_tokens]            FP32 contiguous.
    """
    assert x_flat.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert x_flat.dim() == 2 and fn.dim() == 2
    assert x_flat.shape[1] == fn.shape[1], (
        f"K mismatch: x_flat.shape[1]={x_flat.shape[1]} vs fn.shape[1]={fn.shape[1]}"
    )

    import deep_gemm

    num_tokens, _ = x_flat.shape
    hc_mult3, _ = fn.shape

    gemm_out = torch.empty(
        num_tokens, hc_mult3, dtype=torch.float32, device=x_flat.device,
    )
    sqrsum = torch.empty(
        num_tokens, dtype=torch.float32, device=x_flat.device,
    )

    deep_gemm.tf32_hc_prenorm_gemm(x_flat, fn, gemm_out, sqrsum)
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
    n_splits_pre: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in deepgemm-backed replacement for `mhc_pre_triton`.

    Same dispatch tree as the Triton path: GEMM via
    `deep_gemm.tf32_hc_prenorm_gemm` instead of the Triton split-K /
    simple kernel; everything else (sinkhorn + per-head normalization
    + output combination) reuses the existing `mhc_pre_big_fuse_triton`
    Triton kernel because deepgemm has no big_fuse equivalent.

    Returns (post_mix [..., hc_mult, 1], comb_mix [..., hc_mult,
    hc_mult], layer_input [..., hidden]).
    """
    # n_splits_pre is part of the Triton split-K wrapper signature; the
    # deepgemm GEMM does not take a comparable knob (it does its own
    # split-K internally when num_splits is unset).
    del n_splits_pre

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
    assert n_splits == 1, "deepgemm path supports n_splits == 1 only"

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

    # Stage 1: GEMM + sum-of-squares via deepgemm.
    x_flat = residual_flat.view(num_tokens, hc_hidden_size)
    gemm_out_2d, gemm_sqr_1d = deepgemm_hc_pre_gemm(x_flat, fn)

    # big_fuse expects [n_splits, n_tokens, hc_mult3] and [n_splits, n_tokens].
    gemm_out_mul = gemm_out_2d.unsqueeze(0)        # [1, n, hc_mult3]
    gemm_out_sqrsum = gemm_sqr_1d.unsqueeze(0)     # [1, n]

    mhc_pre_big_fuse_triton(
        gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual_flat,
        post_mix, comb_mix, layer_input,
        hidden_size=hidden_size,
        rms_eps=rms_eps, hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        n_splits=n_splits, hc_mult=hc_mult,
    )

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)
    return post_mix, comb_mix, layer_input
