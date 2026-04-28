"""Standalone correctness harness for the sm_120 Triton
`fp8_paged_mqa_logits` kernel against the torch reference.

Run on a CUDA box (sm_120 desktop Blackwell or similar) with the V4
service stopped:

    python -m sglang.jit_kernel._test_fp8_paged_mqa_logits

Exits non-zero on the first mismatch.
"""
from __future__ import annotations

import sys

import torch

from sglang.jit_kernel.deepseek_v4 import fp8_paged_mqa_logits_triton
from sglang.srt.layers.attention.compressed.indexer import (
    fp8_paged_mqa_logits_torch,
)


def _build_synthetic_cache(
    num_pages: int,
    block_size: int,
    head_dim: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Build a (num_pages, block_size, 1, head_dim + 4)-shaped uint8 tensor
    populated with split-per-page layout so both the torch reference and
    the new Triton kernel see the same K bytes and scale bytes.

    Layout per page (matching the torch reference's slice contract at
    `compressed/indexer.py:68-84`):

        bytes [0, block_size * head_dim)              : K bytes
        bytes [block_size * head_dim, total_per_page) : scale bytes

    For positions [in_page]:
        K bytes:     [in_page * head_dim, (in_page + 1) * head_dim)
        scale bytes: [scale_base + in_page * 4, scale_base + (in_page + 1) * 4)
    """
    g = torch.Generator(device=device).manual_seed(seed)
    total_per_page = block_size * (head_dim + 4)
    cache_bytes = torch.empty((num_pages, total_per_page), dtype=torch.uint8, device=device)

    # Random FP8-valued K bytes (well-distributed but not extreme).
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    k_fp32 = (
        torch.empty(num_pages, block_size * head_dim, device=device)
        .uniform_(-fp8_max * 0.5, fp8_max * 0.5, generator=g)
    )
    k_fp8 = k_fp32.to(torch.float8_e4m3fn)
    cache_bytes[:, : block_size * head_dim] = k_fp8.view(torch.uint8)

    # Random fp32 scales in a reasonable range.
    scales_fp32 = (
        torch.empty(num_pages, block_size, device=device)
        .uniform_(0.1, 2.0, generator=g)
    )
    scales_bytes = scales_fp32.view(torch.uint8).reshape(num_pages, block_size * 4)
    cache_bytes[:, block_size * head_dim :] = scales_bytes

    return cache_bytes.view(num_pages, block_size, 1, head_dim + 4)


def _run_case(
    name: str,
    *,
    B: int,
    num_heads: int,
    num_pages: int,
    max_pages_per_req: int,
    seq_lens_list: list[int],
    max_seq_len: int,
    inject_negative_pages: bool,
    device: torch.device,
    seed: int,
    rtol: float = 5e-2,
    atol: float = 5e-2,
) -> bool:
    head_dim = 128
    block_size = 64
    assert len(seq_lens_list) == B

    g = torch.Generator(device=device).manual_seed(seed)

    kvcache = _build_synthetic_cache(num_pages, block_size, head_dim, device, seed)

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    q_fp32 = (
        torch.empty(B, 1, num_heads, head_dim, device=device)
        .uniform_(-fp8_max * 0.5, fp8_max * 0.5, generator=g)
    )
    q_fp8 = q_fp32.to(torch.float8_e4m3fn)

    weight = (
        torch.empty(B, num_heads, dtype=torch.float32, device=device)
        .uniform_(0.0, 1.0, generator=g)
    )
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

    page_table = torch.randint(
        low=0,
        high=num_pages,
        size=(B, max_pages_per_req),
        generator=g,
        device=device,
        dtype=torch.int32,
    )
    if inject_negative_pages:
        # Replace pages past each batch's seq_len with -1, exercising the
        # `pages_clamped = page_table.clamp(min=0)` branch in the torch
        # reference.
        for b, sl in enumerate(seq_lens_list):
            first_invalid = (sl + block_size - 1) // block_size
            if first_invalid < max_pages_per_req:
                page_table[b, first_invalid:] = -1

    out_torch = fp8_paged_mqa_logits_torch(
        q_fp8.clone(),
        kvcache.clone(),
        weight.clone(),
        seq_lens.clone(),
        page_table.clone(),
        deep_gemm_metadata=None,
        max_seq_len=max_seq_len,
        clean_logits=False,
    )
    out_triton = fp8_paged_mqa_logits_triton(
        q_fp8,
        kvcache,
        weight,
        seq_lens,
        page_table,
        deep_gemm_metadata=None,
        max_seq_len=max_seq_len,
        clean_logits=False,
    )

    if out_torch.shape != out_triton.shape:
        print(f"[FAIL] {name}: shape mismatch torch={out_torch.shape} triton={out_triton.shape}")
        return False

    diff = (out_torch - out_triton).abs()
    max_abs = diff.max().item()
    max_rel = (diff / out_torch.abs().clamp(min=1e-6)).max().item()
    ok = torch.allclose(out_torch, out_triton, rtol=rtol, atol=atol)

    status = "OK  " if ok else "FAIL"
    print(
        f"[{status}] {name}: max_abs={max_abs:.4g} max_rel={max_rel:.4g} "
        f"shape={tuple(out_torch.shape)}"
    )
    if not ok:
        # Print one offending row
        bad_b, bad_p = (diff > atol + rtol * out_torch.abs()).nonzero()[0].tolist()
        print(
            f"       first mismatch at b={bad_b}, p={bad_p}: "
            f"torch={out_torch[bad_b, bad_p].item()} triton={out_triton[bad_b, bad_p].item()}"
        )
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable; skipping.")
        return 0
    device = torch.device("cuda:0")

    cases = [
        dict(
            name="B=1 heads=64 seq=200 pages=4",
            B=1, num_heads=64, num_pages=64, max_pages_per_req=4,
            seq_lens_list=[200], max_seq_len=256,
            inject_negative_pages=False, seed=1,
        ),
        dict(
            name="B=2 heads=128 seq=odd pages=8",
            B=2, num_heads=128, num_pages=128, max_pages_per_req=8,
            seq_lens_list=[123, 511], max_seq_len=512,
            inject_negative_pages=False, seed=2,
        ),
        dict(
            name="B=4 heads=32 mixed seqs with -1 pages",
            B=4, num_heads=32, num_pages=64, max_pages_per_req=6,
            seq_lens_list=[64, 100, 256, 380], max_seq_len=384,
            inject_negative_pages=True, seed=3,
        ),
        dict(
            name="B=1 heads=128 max_seq_len > padded_seq_len",
            B=1, num_heads=128, num_pages=32, max_pages_per_req=2,
            seq_lens_list=[100], max_seq_len=300,  # padded=128 < 300
            inject_negative_pages=False, seed=4,
        ),
        dict(
            name="B=1 heads=16 seq divisible by 64",
            B=1, num_heads=16, num_pages=16, max_pages_per_req=4,
            seq_lens_list=[256], max_seq_len=256,
            inject_negative_pages=False, seed=5,
        ),
    ]

    failures = 0
    for case in cases:
        if not _run_case(device=device, **case):
            failures += 1

    if failures:
        print(f"\n{failures}/{len(cases)} cases FAILED")
        return 1
    print(f"\nAll {len(cases)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
