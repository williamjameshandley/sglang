"""Standalone correctness harness for the sm_120 Triton sparse-MLA decode
kernel against:

  Oracle A — Phase 6.1 pure-PyTorch reference
             (`_sparse_mla_torch_reference.flash_mla_with_kvcache_torch_reference`)
  Oracle B — existing live torch fallback
             (`debug_flash_mla_adapter.flash_mla_with_kvcache_torch`)

Per the Phase 6.1 LSE contract, Oracle A and Oracle B share identical
semantics (plain logsumexp + sink-scaled output + lonely-query +inf
correction), so this harness validates that:

  Triton ≈ Oracle A    AND    Oracle A == Oracle B

Run on a CUDA box (sm_120) with the V4 service stopped:

    python -m sglang.jit_kernel._test_sparse_mla

Exits non-zero on the first failure.
"""
from __future__ import annotations

import sys
from typing import Optional

import torch

from sglang.jit_kernel._sparse_mla_torch_reference import (
    flash_mla_with_kvcache_torch_reference,
)
from sglang.jit_kernel.deepseek_v4 import flash_mla_with_kvcache_triton_sm120
from sglang.srt.flashmla_tests import quant as flashmla_quant


def _build_quantized_cache(
    num_pages: int,
    P: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Build a [num_pages, P, 1, 584] uint8 quantized K cache matching the
    layout `DeepSeekV4SingleKVPool` produces and `flashmla_quant` writes.

    Returns the cache as ``uint8`` with ``stride(0) == bytes_per_page_padded``
    (a multiple of 576 ≥ P*584), as the live call-site reshape produces.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    # Synthetic K with bounded magnitude so UE8M0 scales stay in mid-range
    # bytes (~110-130) — avoids edge bytes (0, 255 = NaN) that
    # `flashmla_quant` may treat specially.
    k_fp = torch.empty((num_pages, P, 1, 512), dtype=torch.bfloat16, device=device)
    k_fp.uniform_(-2.0, 2.0, generator=g)

    # Use the canonical quantizer to produce the per-page-split byte layout.
    quantized = flashmla_quant.quantize_k_cache(
        k_fp, flashmla_quant.FP8KVCacheLayout.MODEL1_FP8Sparse,
    )
    # `quantize_k_cache` returns shape [num_pages, P, 1, 584] but its
    # underlying allocation may not have stride(0) % 576 == 0.
    # Re-allocate a properly-padded buffer matching `create_buffer` and copy.
    bytes_per_token = 584
    bytes_per_page_padded = ((P * bytes_per_token + 575) // 576) * 576
    padded = torch.zeros(
        (num_pages, bytes_per_page_padded),
        dtype=torch.uint8,
        device=device,
    )
    quantized_u8 = quantized.view(flashmla_quant.FP8_DTYPE).view(torch.uint8)
    padded[:, : P * bytes_per_token] = quantized_u8.reshape(
        num_pages, P * bytes_per_token,
    )
    return padded[:, : P * bytes_per_token].view(num_pages, P, 1, bytes_per_token)


def _gen_indices(
    B: int,
    topk: int,
    num_pages: int,
    P: int,
    topk_lengths: list[int],
    sprinkle_neg1_within_topk_length: bool,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Generate flat indices ``[B, 1, topk]`` int32.

    Values drawn from ``{-1} ∪ [0, num_pages*P)``; never positive
    out-of-range (matches `get_swa_page_indices` contract). Optionally
    sprinkles ``-1`` entries within each batch's topk_length region to
    exercise the kernel's logit-mask path for invalid token IDs.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    max_flat = num_pages * P
    indices = torch.randint(
        0, max_flat, (B, 1, topk), generator=g, device=device, dtype=torch.int32,
    )
    for b in range(B):
        tl = topk_lengths[b]
        # Tail beyond tl: -1 (matches the live `swa_page_indices` contract)
        if tl < topk:
            indices[b, 0, tl:] = -1
        if sprinkle_neg1_within_topk_length and tl > 4:
            # Sprinkle a few -1 entries at known positions inside [0, tl)
            stride = max(1, tl // 4)
            for k in range(0, tl, stride):
                indices[b, 0, k] = -1
    return indices


def _close(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> tuple[bool, float, float]:
    diff = (a.float() - b.float()).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    denom = b.float().abs().clamp(min=1e-6)
    max_rel = (diff / denom).max().item() if diff.numel() else 0.0
    ok = torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)
    return ok, max_abs, max_rel


def _run_case(
    name: str,
    *,
    B: int,
    h_q: int,
    P: int,
    num_pages: int,
    topk: int,
    topk_lengths: list[int],
    sprinkle_neg1: bool,
    sink_zeros: bool,
    device: torch.device,
    seed: int,
    # Phase 6.7 optional compressed scope
    P_extra: int = 0,
    extra_num_pages: int = 0,
    extra_topk: int = 0,
    extra_topk_lengths: Optional[list[int]] = None,
    extra_sprinkle_neg1: bool = False,
) -> bool:
    assert len(topk_lengths) == B
    assert topk % 64 == 0
    assert h_q % 16 == 0
    has_extra = P_extra > 0
    if has_extra:
        assert extra_topk_lengths is not None and len(extra_topk_lengths) == B
        assert extra_topk % 64 == 0

    g = torch.Generator(device=device).manual_seed(seed + 7)

    k_cache = _build_quantized_cache(num_pages, P, device, seed=seed)
    bytes_per_page_padded = k_cache.stride(0)
    if bytes_per_page_padded % 576 != 0:
        print(f"[FAIL] {name}: pool stride {bytes_per_page_padded} not %576==0")
        return False

    q = torch.empty((B, 1, h_q, 512), dtype=torch.bfloat16, device=device)
    q.uniform_(-1.0, 1.0, generator=g)

    indices = _gen_indices(
        B, topk, num_pages, P, topk_lengths, sprinkle_neg1, device, seed=seed,
    )
    topk_length = torch.tensor(topk_lengths, dtype=torch.int32, device=device)

    extra_k_cache = None
    extra_indices = None
    extra_topk_length = None
    if has_extra:
        extra_k_cache = _build_quantized_cache(
            extra_num_pages, P_extra, device, seed=seed + 100,
        )
        if extra_k_cache.stride(0) % 576 != 0:
            print(f"[FAIL] {name}: extra pool stride not %576==0")
            return False
        extra_indices = _gen_indices(
            B, extra_topk, extra_num_pages, P_extra,
            extra_topk_lengths, extra_sprinkle_neg1, device, seed=seed + 200,
        )
        extra_topk_length = torch.tensor(
            extra_topk_lengths, dtype=torch.int32, device=device,
        )

    if sink_zeros:
        attn_sink = torch.zeros(h_q, dtype=torch.float32, device=device)
    else:
        attn_sink = torch.empty(h_q, dtype=torch.float32, device=device)
        attn_sink.uniform_(-2.0, 2.0, generator=g)

    common_kwargs = dict(
        q=q,
        k_cache=k_cache,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        softmax_scale=512 ** -0.5,
        head_dim_v=512,
        is_fp8_kvcache=True,
        causal=False,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_topk_length,
    )

    out_a, lse_a = flash_mla_with_kvcache_torch_reference(**common_kwargs)
    pack_b = _call_oracle_b(common_kwargs)
    if pack_b is None:
        return False
    out_b, lse_b = pack_b
    out_t, lse_t = flash_mla_with_kvcache_triton_sm120(**common_kwargs)

    # Oracle A vs Oracle B (must agree by construction; if not, oracle bug)
    ok_ab_out, ab_out_abs, ab_out_rel = _close(out_a, out_b, atol=1e-2, rtol=1e-2)
    ok_ab_lse, ab_lse_abs, _ = _close(lse_a, lse_b, atol=1e-3, rtol=1e-3)
    if not (ok_ab_out and ok_ab_lse):
        print(
            f"[FAIL] {name}: Oracle A vs Oracle B disagree "
            f"(out_abs={ab_out_abs:.3g} lse_abs={ab_lse_abs:.3g}) — "
            f"likely oracle bug, not Triton bug"
        )
        return False

    # Triton vs Oracle A
    ok_out, out_abs, out_rel = _close(out_t, out_a, atol=5e-2, rtol=5e-2)
    ok_lse, lse_abs, lse_rel = _close(lse_t, lse_a, atol=5e-2, rtol=5e-2)

    status = "OK  " if (ok_out and ok_lse) else "FAIL"
    print(
        f"[{status}] {name}: out_abs={out_abs:.3g} out_rel={out_rel:.3g} "
        f"lse_abs={lse_abs:.3g} lse_rel={lse_rel:.3g}"
    )
    if not (ok_out and ok_lse):
        # Show first mismatched (b, h) pair
        diff = (out_t.float() - out_a.float()).abs().max(dim=-1).values
        bad = (diff > 5e-2).nonzero()
        if bad.numel() > 0:
            b, sq, h = bad[0].tolist()
            print(
                f"       first out mismatch at (b={b}, s_q={sq}, h={h}): "
                f"triton={out_t[b,sq,h,:5].tolist()} oracle={out_a[b,sq,h,:5].tolist()}"
            )
        return False
    return True


def _call_oracle_b(kwargs: dict) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Run the existing torch adapter (Oracle B). Mirrors its arg shape.

    The adapter doesn't accept a `head_dim_v` kwarg the same way; thread
    only the kwargs it expects.
    """
    from sglang.srt.layers.attention.debug_flash_mla_adapter import (
        flash_mla_with_kvcache_torch,
    )
    try:
        return flash_mla_with_kvcache_torch(
            q=kwargs["q"],
            k_cache=kwargs["k_cache"],
            block_table=None,
            cache_seqlens=None,
            head_dim_v=kwargs["head_dim_v"],
            tile_scheduler_metadata=None,
            num_splits=None,
            softmax_scale=kwargs["softmax_scale"],
            causal=False,
            is_fp8_kvcache=True,
            indices=kwargs["indices"],
            attn_sink=kwargs["attn_sink"],
            extra_k_cache=kwargs.get("extra_k_cache"),
            extra_indices_in_kvcache=kwargs.get("extra_indices_in_kvcache"),
            topk_length=kwargs["topk_length"],
            extra_topk_length=kwargs.get("extra_topk_length"),
        )
    except Exception as e:
        print(f"       Oracle B raised: {type(e).__name__}: {e}")
        return None


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA unavailable; skipping.")
        return 0
    device = torch.device("cuda:0")

    cases = [
        # (radix-mode SWA pool: P=256)
        dict(
            name="B=1 h=64 P=256 topk=64 lens=[40] sink=0",
            B=1, h_q=64, P=256, num_pages=8, topk=64,
            topk_lengths=[40], sprinkle_neg1=False, sink_zeros=True, seed=1,
        ),
        dict(
            name="B=1 h=64 P=256 topk=128 lens=[97] -1 sink!=0",
            B=1, h_q=64, P=256, num_pages=8, topk=128,
            topk_lengths=[97], sprinkle_neg1=True, sink_zeros=False, seed=2,
        ),
        dict(
            name="B=2 h=64 P=256 topk=64 lens=[33,1]",
            B=2, h_q=64, P=256, num_pages=8, topk=64,
            topk_lengths=[33, 1], sprinkle_neg1=False, sink_zeros=False, seed=3,
        ),
        dict(
            name="B=2 h=128 P=256 topk=256 lens=[200,150] -1",
            B=2, h_q=128, P=256, num_pages=16, topk=256,
            topk_lengths=[200, 150], sprinkle_neg1=True, sink_zeros=False, seed=4,
        ),
        dict(
            name="B=2 h=64 P=256 topk=64 lens=[0, 5] (lonely-query)",
            B=2, h_q=64, P=256, num_pages=8, topk=64,
            topk_lengths=[0, 5], sprinkle_neg1=False, sink_zeros=False, seed=5,
        ),
        dict(
            name="B=1 h=128 P=256 topk=512 lens=[400] -1",
            B=1, h_q=128, P=256, num_pages=16, topk=512,
            topk_lengths=[400], sprinkle_neg1=True, sink_zeros=False, seed=6,
        ),
        # Non-radix SWA pool: P=128
        dict(
            name="B=2 h=64 P=128 topk=128 lens=[100,128]",
            B=2, h_q=64, P=128, num_pages=16, topk=128,
            topk_lengths=[100, 128], sprinkle_neg1=False, sink_zeros=False, seed=7,
        ),
        # h_q=16 boundary
        dict(
            name="B=1 h=16 P=256 topk=64 lens=[50]",
            B=1, h_q=16, P=256, num_pages=4, topk=64,
            topk_lengths=[50], sprinkle_neg1=False, sink_zeros=False, seed=8,
        ),
        # topk_length > topk (mask is purely positional)
        dict(
            name="B=1 h=64 P=256 topk=64 lens=[200]>topk",
            B=1, h_q=64, P=256, num_pages=8, topk=64,
            topk_lengths=[200], sprinkle_neg1=False, sink_zeros=False, seed=9,
        ),
        dict(
            name="B=4 h=64 P=256 topk=128 mixed lens",
            B=4, h_q=64, P=256, num_pages=16, topk=128,
            topk_lengths=[64, 100, 128, 33], sprinkle_neg1=True,
            sink_zeros=False, seed=10,
        ),
        # Phase 6.7 compressed-scope cases
        dict(
            name="C4 B=1 h=64 P=256 topk=64 + P_extra=64 etopk=64",
            B=1, h_q=64, P=256, num_pages=8, topk=64,
            topk_lengths=[40], sprinkle_neg1=False, sink_zeros=False, seed=20,
            P_extra=64, extra_num_pages=8, extra_topk=64,
            extra_topk_lengths=[30], extra_sprinkle_neg1=False,
        ),
        dict(
            name="C4 B=2 h=128 P=256 topk=128 + P_extra=64 etopk=128 -1",
            B=2, h_q=128, P=256, num_pages=16, topk=128,
            topk_lengths=[100, 64], sprinkle_neg1=True, sink_zeros=False, seed=21,
            P_extra=64, extra_num_pages=16, extra_topk=128,
            extra_topk_lengths=[80, 50], extra_sprinkle_neg1=True,
        ),
        dict(
            name="C128 B=1 h=64 P=256 topk=64 + P_extra=2 etopk=64",
            B=1, h_q=64, P=256, num_pages=8, topk=64,
            topk_lengths=[50], sprinkle_neg1=False, sink_zeros=False, seed=22,
            P_extra=2, extra_num_pages=64, extra_topk=64,
            extra_topk_lengths=[40], extra_sprinkle_neg1=False,
        ),
        dict(
            name="C4 lonely both scopes (B=1 lens=0 elens=0)",
            B=1, h_q=64, P=256, num_pages=8, topk=64,
            topk_lengths=[0], sprinkle_neg1=False, sink_zeros=False, seed=23,
            P_extra=64, extra_num_pages=8, extra_topk=64,
            extra_topk_lengths=[0], extra_sprinkle_neg1=False,
        ),
        dict(
            name="C4 only-compressed-valid (lens=0 elens>0)",
            B=1, h_q=64, P=256, num_pages=8, topk=64,
            topk_lengths=[0], sprinkle_neg1=False, sink_zeros=False, seed=24,
            P_extra=64, extra_num_pages=8, extra_topk=64,
            extra_topk_lengths=[40], extra_sprinkle_neg1=False,
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
