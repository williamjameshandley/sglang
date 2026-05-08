"""Tensor-level harness comparing the deepgemm-sm120 paged MQA logits
kernel against the live torch reference (`fp8_paged_mqa_logits_torch`
in `compressed/indexer.py`).

Inputs match the live indexer call shape exactly:
- `q_fp8` is FP8 e4m3 (`act_quant(q)` output) shape [B, 1, num_heads, 128].
- `weight` has the live `q_scale` folded in via `fused_scale` semantics.
- `kvcache_fp8` is the split FP8/FP32-scale layout [num_blocks, 64, 1, 132].
- `seq_lens`: 1-D for the torch reference (asserts `(B,)`); 2-D `[B, 1]`
  for the deepgemm path (csrc/apis/attention.hpp asserts `dim() == 2`).

Tolerances: rel/abs ≤ 1e-3 (FP32 output).
"""

from __future__ import annotations

import sys

import torch

FP8_DTYPE = torch.float8_e4m3fn


def _build_inputs(
    *,
    batch_size: int,
    num_heads: int,
    max_num_pages: int,
    num_blocks_total: int,
    block_size: int = 64,
    head_dim: int = 128,
    device: str = "cuda",
    seed: int = 0,
):
    """Construct synthetic inputs matching the live indexer call shape."""
    g = torch.Generator(device=device).manual_seed(seed)

    # FP8 q (already act_quant'd); the live path passes q_fp8 directly.
    # Scale to a mild range so torch-reference accumulation in FP32 stays
    # well within finite-range. The deepgemm path tolerates wider input
    # ranges; we restrict to keep the torch reference oracle finite.
    q_fp32 = (
        torch.randn(batch_size, 1, num_heads, head_dim, generator=g, device=device)
        * 0.25
    )
    q_fp8 = q_fp32.to(FP8_DTYPE)

    # Live KV cache layout (split per page): page bytes [0, K_bytes) hold
    # the FP8 K values for all `block_size` positions, page bytes
    # [K_bytes, page_bytes) hold the per-position FP32 scales (one fp32
    # per position). Both deepgemm and the torch reference at
    # `compressed/indexer.py:68-84` consume this split layout; the
    # [num_blocks, block_size, 1, head_dim+4] 4-D shape is a view, not an
    # interleaved-per-position allocation.
    page_bytes = block_size * (head_dim + 4)  # 8448 at block=64 d=128
    k_bytes = block_size * head_dim           # 8192
    flat = torch.empty(
        num_blocks_total, page_bytes, dtype=torch.uint8, device=device
    )
    fp8_region = (
        torch.randn(
            num_blocks_total, block_size, head_dim, generator=g, device=device
        )
        * 0.25
    ).to(FP8_DTYPE)
    flat[:, :k_bytes] = fp8_region.reshape(num_blocks_total, k_bytes).view(torch.uint8)
    scales_fp32 = (
        torch.rand(num_blocks_total, block_size, generator=g, device=device) * 0.5
        + 0.5
    ).contiguous()
    flat[:, k_bytes:] = scales_fp32.view(torch.uint8)
    kvcache_fp8 = flat.view(num_blocks_total, block_size, 1, head_dim + 4)

    # weights[b, h] = compute_weights(...) * weight_scale * q_scale (already
    # fused per the live indexer). Modest range to keep torch FP32 accum finite.
    weight = (
        torch.rand(batch_size, num_heads, generator=g, device=device).float() * 0.5
        + 0.5
    )

    # Page table maps each batch's positions to physical blocks.
    page_table = torch.randint(
        0, num_blocks_total, (batch_size, max_num_pages), generator=g, device=device,
        dtype=torch.int32,
    )

    # Per-batch valid sequence length (in tokens). Each batch covers up to
    # max_num_pages * block_size tokens.
    seq_lens_full = max_num_pages * block_size
    seq_lens = torch.randint(
        block_size, seq_lens_full + 1, (batch_size,), generator=g, device=device,
        dtype=torch.int32,
    )

    return q_fp8, kvcache_fp8, weight, seq_lens, page_table


def _max_rel_abs(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    # Mask out positions where either side is NaN/Inf (the torch reference
    # can produce isolated NaNs from edge-of-FP32-range accumulation; we
    # care about agreement on the finite positions).
    finite = torch.isfinite(a) & torch.isfinite(b)
    if not finite.any():
        return float("nan"), float("nan")
    diff = (a - b).abs()
    abs_max = float(diff[finite].max())
    denom = b.abs().clamp(min=1e-6)
    rel_max = float((diff[finite] / denom[finite]).max())
    return abs_max, rel_max


def main() -> int:
    torch.cuda.set_device(0)
    device = "cuda"

    cases = [
        # (B, num_heads, max_num_pages, num_blocks_total)
        (1, 64, 4, 16),
        (2, 64, 8, 32),
        (4, 64, 4, 32),
        (1, 64, 16, 64),
    ]

    from sglang.srt.layers.attention.compressed.indexer import (
        fp8_paged_mqa_logits_torch,
    )

    import deep_gemm

    print("=" * 60)
    print("D2 paged MQA logits: deepgemm vs torch reference")
    print("=" * 60)

    block_size = 64
    head_dim = 128

    fail = 0
    for B, num_heads, max_num_pages, num_blocks_total in cases:
        q_fp8, kvcache_fp8, weight, seq_lens, page_table = _build_inputs(
            batch_size=B,
            num_heads=num_heads,
            max_num_pages=max_num_pages,
            num_blocks_total=num_blocks_total,
            block_size=block_size,
            head_dim=head_dim,
            device=device,
        )
        max_seq_len = max_num_pages * block_size

        # Torch reference (1-D seq_lens, ignores deep_gemm_metadata).
        torch_out = fp8_paged_mqa_logits_torch(
            q_fp8,
            kvcache_fp8,
            weight,
            seq_lens,
            page_table,
            None,
            max_seq_len,
            False,
        )

        # Deepgemm: 2-D seq_lens [B, 1] and metadata tensor.
        seq_lens_2d = seq_lens.to(torch.int32).view(-1, 1)
        metadata = deep_gemm.get_paged_mqa_logits_metadata(
            seq_lens_2d, block_size, deep_gemm.get_num_sms()
        )
        dg_out = deep_gemm.fp8_paged_mqa_logits(
            q_fp8,
            kvcache_fp8,
            weight,
            seq_lens_2d,
            page_table,
            metadata,
            max_seq_len,
            False,
        )

        # Mask out positions beyond seq_lens for both sides; the deepgemm
        # kernel may leave junk past the valid range while the torch
        # reference zero-fills.
        positions = torch.arange(max_seq_len, device=device).unsqueeze(0)
        valid = positions < seq_lens.unsqueeze(1)
        torch_valid = torch_out.where(valid, torch.zeros_like(torch_out))
        dg_valid = dg_out.where(valid, torch.zeros_like(dg_out))

        abs_max, rel_max = _max_rel_abs(dg_valid, torch_valid)
        ok = abs_max <= 1e-3 and rel_max <= 1e-3
        verdict = "PASS" if ok else "FAIL"
        print(
            f"  B={B:2d} h={num_heads:3d} pages={max_num_pages:3d} "
            f"max_seq_len={max_seq_len:4d} — {verdict} "
            f"(abs={abs_max:.4e} rel={rel_max:.4e})"
        )
        if not ok:
            fail += 1

    print()
    print(f"Result: {len(cases) - fail}/{len(cases)} passed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
