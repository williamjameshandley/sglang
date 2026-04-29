"""Correctness harness for the V4-Flash MHC Triton kernels.

Phase 7.4: validates each Triton MHC kernel against the torch fallback
oracle. The torch fallbacks live in `python/sglang/srt/models/deepseek_v4.py`
(`hc_pre_torch_impl`, `hc_post_torch_impl`); this harness implements
the same formulas inline so it doesn't depend on importing the V4
model.

Run on a CUDA box:

    python -m sglang.jit_kernel._test_mhc_triton

Exits non-zero on the first mismatch.
"""
from __future__ import annotations

import sys
from typing import Tuple

import torch
import torch.nn.functional as F

from sglang.jit_kernel.mhc_triton import (
    mhc_post_triton,
    mhc_pre_gemm_sqrsum_splitk_triton,
    mhc_pre_gemm_sqrsum_triton,
)


def _oracle_mhc_post(x, residual, post, comb):
    """Reference: matches `hc_post_torch_impl` in
    `models/deepseek_v4.py:1875-1880`."""
    return (
        post.unsqueeze(-1) * x.unsqueeze(1)
        + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
    ).type_as(x)


def _run_mhc_post_case(
    name: str,
    *,
    num_tokens: int,
    hc: int,
    hidden: int,
    device: torch.device,
    seed: int,
) -> bool:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=device)
    residual = torch.empty((num_tokens, hc, hidden), dtype=torch.bfloat16, device=device)
    post = torch.empty((num_tokens, hc), dtype=torch.float32, device=device)
    comb = torch.empty((num_tokens, hc, hc), dtype=torch.float32, device=device)
    if num_tokens > 0:
        x.uniform_(-0.5, 0.5, generator=g)
        residual.uniform_(-0.5, 0.5, generator=g)
        post.uniform_(-1.0, 1.0, generator=g)
        comb.uniform_(-1.0, 1.0, generator=g)

    out_t = mhc_post_triton(x, residual, post, comb)
    out_o = _oracle_mhc_post(x, residual, post, comb)

    if num_tokens == 0:
        if out_t.shape != (0, hc, hidden):
            print(f"[FAIL] {name}: empty shape {tuple(out_t.shape)}")
            return False
        print(f"[OK  ] {name}: empty fast path")
        return True

    ok, abs_, rel = _close(out_t, out_o, atol=5e-2, rtol=5e-2)
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] {name}: abs={abs_:.3g} rel={rel:.3g}")
    return ok


def _oracle_pre_gemm_sqrsum(
    x: torch.Tensor,            # [num_tokens, hc_hidden] bf16
    fn: torch.Tensor,            # [hc_mult3, hc_hidden] fp32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference: `out = F.linear(x, fn)`, `sqrsum = (x*x).sum(-1)`.

    This matches the torch fallback path's GEMM + sqrsum semantics for
    the MHC pre block. fp32 accumulation throughout.
    """
    x_f32 = x.to(torch.float32)
    out = F.linear(x_f32, fn)                  # [N, hc_mult3]
    sqrsum = (x_f32 * x_f32).sum(dim=-1)       # [N]
    return out, sqrsum


def _close(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float):
    diff = (a.float() - b.float()).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    denom = b.float().abs().clamp(min=1e-6)
    max_rel = (diff / denom).max().item() if diff.numel() else 0.0
    ok = torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)
    return ok, max_abs, max_rel


def _run_pre_gemm_sqrsum_case(
    name: str,
    *,
    num_tokens: int,
    hc_mult3: int,
    hc_hidden: int,
    n_splits_pre: int,
    device: torch.device,
    seed: int,
    use_simple: bool = False,
) -> bool:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.empty((num_tokens, hc_hidden), dtype=torch.bfloat16, device=device)
    if num_tokens > 0:
        x.uniform_(-0.5, 0.5, generator=g)
    fn = torch.empty((hc_mult3, hc_hidden), dtype=torch.float32, device=device)
    fn.uniform_(-0.05, 0.05, generator=g)

    if use_simple:
        out_t, sqr_t = mhc_pre_gemm_sqrsum_triton(x, fn, hc_mult3=hc_mult3)
    else:
        out_t, sqr_t = mhc_pre_gemm_sqrsum_splitk_triton(
            x, fn, hc_mult3=hc_mult3, n_splits_pre=n_splits_pre,
        )
    out_o, sqr_o = _oracle_pre_gemm_sqrsum(x, fn)

    if num_tokens == 0:
        # Empty-input fast path: shape parity + no-launch.
        if out_t.shape != (0, hc_mult3) or sqr_t.shape != (0,):
            print(f"[FAIL] {name}: empty shape mismatch "
                  f"out={tuple(out_t.shape)} sqr={tuple(sqr_t.shape)}")
            return False
        print(f"[OK  ] {name}: empty fast path")
        return True

    ok_out, out_abs, out_rel = _close(out_t, out_o, atol=5e-2, rtol=5e-2)
    ok_sqr, sqr_abs, sqr_rel = _close(sqr_t, sqr_o, atol=5e-2, rtol=5e-2)
    status = "OK  " if (ok_out and ok_sqr) else "FAIL"
    print(
        f"[{status}] {name}: out_abs={out_abs:.3g} out_rel={out_rel:.3g} "
        f"sqr_abs={sqr_abs:.3g} sqr_rel={sqr_rel:.3g}"
    )
    return ok_out and ok_sqr


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable; skipping.")
        return 0
    device = torch.device("cuda:0")

    # V4-Flash uses hc_mult=4 → hc_mult3 = 4*(2+4) = 24.
    # V4-Flash hidden_size=4096 (per its config.json), so MHC pre operates
    # on hc * hidden_size = 4 * 4096 = 16384.
    HC_MULT3 = 24
    HC_HIDDEN = 16384

    cases = [
        # Empty-input fast path
        dict(name="num_tokens=0 (empty)",
             num_tokens=0, hc_mult3=HC_MULT3, hc_hidden=HC_HIDDEN,
             n_splits_pre=32, seed=1),
        # Brackets the split-K threshold (deployment uses split-K for
        # num_tokens <= 2048 per `mhc.py:552-574`).
        dict(name="num_tokens=1",
             num_tokens=1, hc_mult3=HC_MULT3, hc_hidden=HC_HIDDEN,
             n_splits_pre=32, seed=2),
        dict(name="num_tokens=32",
             num_tokens=32, hc_mult3=HC_MULT3, hc_hidden=HC_HIDDEN,
             n_splits_pre=32, seed=3),
        dict(name="num_tokens=128",
             num_tokens=128, hc_mult3=HC_MULT3, hc_hidden=HC_HIDDEN,
             n_splits_pre=32, seed=4),
        dict(name="num_tokens=2048 (boundary)",
             num_tokens=2048, hc_mult3=HC_MULT3, hc_hidden=HC_HIDDEN,
             n_splits_pre=32, seed=5),
        # Smaller split_k variants
        dict(name="num_tokens=128 split_k=16",
             num_tokens=128, hc_mult3=HC_MULT3, hc_hidden=HC_HIDDEN,
             n_splits_pre=16, seed=6),
        dict(name="num_tokens=128 split_k=8",
             num_tokens=128, hc_mult3=HC_MULT3, hc_hidden=HC_HIDDEN,
             n_splits_pre=8, seed=7),
        # Simple (non-split-K) path: num_tokens > 2048 hits this on the
        # live deployment per `mhc.py:575-586`.
        dict(name="simple num_tokens=0",
             num_tokens=0, hc_mult3=HC_MULT3, hc_hidden=HC_HIDDEN,
             n_splits_pre=1, seed=10, use_simple=True),
        dict(name="simple num_tokens=2049",
             num_tokens=2049, hc_mult3=HC_MULT3, hc_hidden=HC_HIDDEN,
             n_splits_pre=1, seed=11, use_simple=True),
        dict(name="simple num_tokens=4096",
             num_tokens=4096, hc_mult3=HC_MULT3, hc_hidden=HC_HIDDEN,
             n_splits_pre=1, seed=12, use_simple=True),
    ]

    failures = 0
    for case in cases:
        if not _run_pre_gemm_sqrsum_case(device=device, **case):
            failures += 1

    # mhc_post cases (V4-Flash hc=4, hidden=4096)
    post_cases = [
        dict(name="post num_tokens=0", num_tokens=0, hc=4, hidden=4096, seed=20),
        dict(name="post num_tokens=1", num_tokens=1, hc=4, hidden=4096, seed=21),
        dict(name="post num_tokens=8", num_tokens=8, hc=4, hidden=4096, seed=22),
        dict(name="post num_tokens=128", num_tokens=128, hc=4, hidden=4096, seed=23),
        dict(name="post num_tokens=2048", num_tokens=2048, hc=4, hidden=4096, seed=24),
    ]
    for case in post_cases:
        if not _run_mhc_post_case(device=device, **case):
            failures += 1

    total = len(cases) + len(post_cases)
    if failures:
        print(f"\n{failures}/{total} cases FAILED")
        return 1
    print(f"\nAll {total} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
