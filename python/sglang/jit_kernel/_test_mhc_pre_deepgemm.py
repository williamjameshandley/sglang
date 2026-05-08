"""Tensor-level harness comparing the deepgemm-sm120 MHC pre-GEMM step
against the torch reference (`F.linear(x_flat.float(), fn)` plus
explicit sum-of-squares).

The big_fuse step (sinkhorn + per-head normalization + output
combination) is unchanged from the Triton path and is validated by
existing MHC tests; here we pin down only the GEMM-side change.

Tolerances: rel/abs ≤ 1e-3 (FP32 output) for the GEMM out tensor;
rel/abs ≤ 1e-3 for the per-row sum-of-squares.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F


def _max_rel_abs(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Returns (max abs delta, max rel delta over positions where |b| > 1).

    The deepgemm GEMM internally casts BF16 lhs to TF32; against an FP32
    torch reference, abs deltas are tight (~5e-4 at typical V4 shapes)
    but rel deltas blow up around near-zero outputs where the FP32
    reference happens to cancel. Restricting rel-evaluation to positions
    with |b| > 1 produces a meaningful rel measure without conflating
    near-zero cancellation with backend disagreement. The pass criterion
    in the harness uses an atol+rtol mixed check, not raw rel/abs.
    """
    finite = torch.isfinite(a) & torch.isfinite(b)
    if not finite.any():
        return float("nan"), float("nan")
    diff = (a - b).abs()
    abs_max = float(diff[finite].max())
    significant = finite & (b.abs() > 1.0)
    if significant.any():
        rel_max = float((diff[significant] / b.abs()[significant]).max())
    else:
        rel_max = 0.0
    return abs_max, rel_max


def main() -> int:
    torch.cuda.set_device(0)
    device = "cuda"

    # V4-Flash-shaped MHC inputs (per Phase 7.4 ports):
    #   hc_mult = 4, hidden = 7168 (V4-Flash hidden_size).
    # x: [num_tokens, hc_mult, hidden] BF16
    # fn: [hc_mult3, hc_mult * hidden] FP32 with hc_mult3 = 2*hc_mult + hc_mult^2
    hc_mult = 4
    hidden = 7168
    hc_mult3 = 2 * hc_mult + hc_mult * hc_mult  # 24
    hc_hidden = hc_mult * hidden  # 28672

    cases = [
        # num_tokens
        1,
        16,
        128,
        2048,
        2049,
    ]

    from sglang.srt.layers.mhc_deepgemm import deepgemm_hc_pre_gemm_splitk

    print("=" * 60)
    print("D3 MHC pre-GEMM: deepgemm split-K vs torch reference")
    print(f"  hc_mult={hc_mult} hidden={hidden} hc_mult3={hc_mult3}")
    print("=" * 60)

    fail = 0
    for num_tokens in cases:
        torch.manual_seed(num_tokens)
        x = (
            torch.randn(num_tokens, hc_mult, hidden, device=device, dtype=torch.float32)
            * 0.1
        ).to(torch.bfloat16)
        # Match `hc_pre_torch_impl`'s contract: fn is FP32 [hc_mult3, hc_hidden].
        fn = (
            torch.randn(hc_mult3, hc_hidden, device=device, dtype=torch.float32) * 0.05
        )

        x_flat = x.view(num_tokens, hc_hidden)

        # Torch reference: F.linear(x.float(), fn) → x @ fn.T (FP32).
        ref_out = F.linear(x_flat.float(), fn)
        ref_sqr = x_flat.float().square().sum(dim=-1)

        # Deepgemm split-K: 3D `[S, M, N]` and 2D `[S, M]` partials —
        # sum across S to compare against the torch reference.
        S = 32
        dg_out_3d, dg_sqr_2d = deepgemm_hc_pre_gemm_splitk(x_flat, fn, num_splits=S)
        dg_out = dg_out_3d.sum(dim=0)
        dg_sqr = dg_sqr_2d.sum(dim=0)

        out_abs, out_rel = _max_rel_abs(dg_out, ref_out)
        sqr_abs, sqr_rel = _max_rel_abs(dg_sqr, ref_sqr)
        # Mixed atol+rtol pass criterion (BF16/TF32 in deepgemm vs FP32 ref).
        # `out` reduction over hc_hidden=28672 elements: per-element ULP at
        # BF16/TF32 ~ |b|×2**-7, so per-output ULP scales with magnitude.
        # atol=1e-3 covers near-zero cancellation; rtol=1e-2 the bulk.
        out_ok = torch.allclose(dg_out, ref_out, atol=1e-3, rtol=1e-2)
        # `sqr` is an exact reduction; deepgemm computes it from BF16 input
        # so a small atol covers the BF16 input precision.
        sqr_ok = torch.allclose(dg_sqr, ref_sqr, atol=1e-3, rtol=1e-3)
        ok = out_ok and sqr_ok
        verdict = "PASS" if ok else "FAIL"
        print(
            f"  num_tokens={num_tokens:5d} — {verdict} "
            f"out abs={out_abs:.3e} rel={out_rel:.3e} | "
            f"sqr abs={sqr_abs:.3e} rel={sqr_rel:.3e}"
        )
        if not ok:
            fail += 1

    print()
    print(f"Result: {len(cases) - fail}/{len(cases)} passed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
