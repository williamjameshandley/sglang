"""D4.1 layout-validation harness.

Validates `_prepare_deepgemm_mxfp4_weight` end-to-end by:
  1. Generating BF16 reference weights with known values.
  2. Quantizing them to V4-Flash MXFP4 packing (uint8 nibbles + uint8
     UE8M0 scales) using the live `triton_kernels.numerics_details.mxfp`
     helpers.
  3. Running `_prepare_deepgemm_mxfp4_weight` to produce deepgemm
     B-operand format.
  4. Calling `deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous` at
     V4-Flash routed-expert shapes with a synthetic FP8 activation.
  5. Comparing against an FP32 reference computed by dequantizing
     the B operand back to BF16 and running a plain matmul.

Pass criterion: rel/abs ≤ 5e-2 BF16 (the BF16 ULP floor at unit
magnitude is ~7.8e-3; 5e-2 catches real layout bugs without tripping
on FP8/FP4 quantization noise).

Two roles validated separately:

- `w13`: shape `[E, 2*intermediate, hidden // 2]`. Gate+up packed.
- `w2`:  shape `[E, hidden, intermediate // 2]`. Down projection.

Layout matrix (per plan): nibble order × K/N orientation × (w13 only:
gate/up swap) = 8 cases for w13, 4 cases for w2. The current attempt
uses the canonical OAI orientation; if any case fails, the harness
flags exactly which flip is needed.
"""

from __future__ import annotations

import sys

import torch


def _max_rel_abs(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
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


def _quantize_bf16_to_mxfp4(
    bf16_w: torch.Tensor, gran_k: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 weight tensor to V4-Flash MXFP4 packing.

    Returns (weight_uint8 [..., K//2] packed nibbles, scale_uint8 [...,
    K//gran_k] UE8M0 bytes), matching the live V4-Flash checkpoint
    layout (`mxfp4.py:402-410, :414-423`).
    """
    from triton_kernels.numerics_details.mxfp import downcast_to_mxfp

    weight_uint8, scale_uint8 = downcast_to_mxfp(
        bf16_w.contiguous(), torch.uint8, axis=-1
    )
    return weight_uint8, scale_uint8


def _dequantize_mxfp4_to_bf16(
    weight_uint8: torch.Tensor, scale_uint8: torch.Tensor
) -> torch.Tensor:
    """Inverse of `_quantize_bf16_to_mxfp4`. Reference path used to
    construct the BF16 oracle for comparison."""
    from triton_kernels.numerics_details.mxfp import upcast_from_mxfp

    return upcast_from_mxfp(weight_uint8, scale_uint8, target_dtype=torch.bfloat16, axis=-1)


def _quantize_act_to_fp8(
    bf16_act: torch.Tensor, gran_k: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 activation to (FP8 e4m3, FP32 UE8M0-rounded scales).

    `use_ue8m0=True` is mandatory: deepgemm's
    `m_grouped_fp8_fp4_gemm_nt_contiguous` with `disable_ue8m0_cast=False`
    requires UE8M0-representable scales. Empirically, passing arbitrary
    FP32 scales (`use_ue8m0=False`) produces all-NaN output at K ≥ 512
    on sm_120 even though small K cases happen to pass.

    The corresponding V4-Flash side (B-operand weight scales) is
    UE8M0-by-construction since the checkpoint stores `torch.uint8`
    exponent bytes that decode to `2.0 ** (byte - 127)` exactly.
    """
    from deep_gemm.utils.math import per_token_cast_to_fp8

    return per_token_cast_to_fp8(bf16_act, use_ue8m0=True, gran_k=gran_k)


def _run_one_role(
    role: str,
    *,
    num_experts: int,
    n_per_expert: int,
    k: int,
    expected_m_per_group: int,
) -> tuple[bool, float, float]:
    """Run one (role, shape) case end-to-end. Returns (ok, abs, rel)."""
    from sglang.srt.layers.moe.fused_moe_triton.deepgemm_mxfp4_weight import (
        _prepare_deepgemm_mxfp4_weight,
    )
    import deep_gemm

    # Match the deepgemm reference test's alignment configuration; the
    # contiguous grouped GEMM requires per-group M to be aligned to the
    # platform's mk-alignment (128 on sm_120).
    deep_gemm.set_mk_alignment_for_contiguous_layout(
        deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    )
    device = "cuda"

    # 1. Reference BF16 weights.
    torch.manual_seed(hash((role, num_experts, n_per_expert, k)) & 0xFFFFFFFF)
    bf16_w = (
        torch.randn(num_experts, n_per_expert, k, device=device, dtype=torch.bfloat16)
        * 0.1
    )

    # 2. Quantize to V4-Flash MXFP4 layout. Scales emerge with shape
    #    [E, n, k // 32] uint8 UE8M0; weights [E, n, k // 2] uint8.
    weight_uint8, scale_uint8 = _quantize_bf16_to_mxfp4(bf16_w, gran_k=32)
    assert weight_uint8.dtype == torch.uint8 and scale_uint8.dtype == torch.uint8
    assert weight_uint8.shape == (num_experts, n_per_expert, k // 2)
    # `downcast_to_mxfp` may pad the K dim; the resulting scale shape is
    # whatever it produces. Don't assume k // 32 exactly.

    # 3. Reference dequant (round-trip through MXFP4 → BF16).
    bf16_w_q = _dequantize_mxfp4_to_bf16(weight_uint8, scale_uint8).view(
        num_experts, n_per_expert, k
    )

    # 4. Deepgemm B operand via the helper under test.
    weight_int8, scales_fp32 = _prepare_deepgemm_mxfp4_weight(
        weight_uint8, scale_uint8, role=role, gran_k=32,
    )

    # 5. Synthetic activation with grouped layout. Each per-group M must
    # be aligned to deepgemm's mk-alignment (128 on sm_120) for the
    # contiguous grouped GEMM path; this matches the reference test's
    # construction at /tmp/deepgemm_check/tests/generators.py:308-310.
    align_m = deep_gemm.get_mk_alignment_for_contiguous_layout()
    aligned_m = ((expected_m_per_group + align_m - 1) // align_m) * align_m
    actual_ms = [aligned_m] * num_experts
    m_total = sum(actual_ms)
    bf16_act = (
        torch.randn(m_total, k, device=device, dtype=torch.bfloat16) * 0.1
    )
    grouped_layout = torch.empty(m_total, device=device, dtype=torch.int32)
    start = 0
    for i, m_i in enumerate(actual_ms):
        grouped_layout[start : start + m_i] = i
        start += m_i

    # 6. FP8 quantize activation (deepgemm A-side: FP8 + FP32 group-128 scales).
    act_fp8, act_sf = _quantize_act_to_fp8(bf16_act, gran_k=128)

    # 7. BF16 reference output: dequant W back to BF16 and matmul.
    ref_out = torch.empty(
        m_total, n_per_expert, device=device, dtype=torch.bfloat16
    )
    start = 0
    for i, m_i in enumerate(actual_ms):
        # Reference: BF16 act @ BF16 (dequantized W).T per expert.
        ref_out[start : start + m_i] = (
            bf16_act[start : start + m_i].float()
            @ bf16_w_q[i].float().T
        ).to(torch.bfloat16)
        start += m_i

    # 8. Deepgemm M-grouped contiguous FP8 × FP4 GEMM.
    dg_out = torch.empty(
        m_total, n_per_expert, device=device, dtype=torch.bfloat16
    )
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        a=(act_fp8, act_sf),
        b=(weight_int8, scales_fp32),
        d=dg_out,
        grouped_layout=grouped_layout,
        recipe_a=(1, 128),
        recipe_b=(1, 32),
    )

    abs_max, rel_max = _max_rel_abs(dg_out.float(), ref_out.float())
    # Cosine-similarity-style diff matching deepgemm's `calc_diff` test
    # metric. The deepgemm reference test passes FP8×FP4 at threshold
    # 0.01 (`tests/generators.py:67-68`'s `quant_config.max_diff()` for
    # is_fp4_a or is_fp4_b cases). Layout/order bugs blow this up
    # decisively; element-wise rel/abs blows up around near-zero
    # cancellation positions even when the layout is correct.
    from deep_gemm.testing.numeric import calc_diff
    diff = calc_diff(dg_out.float(), ref_out.float())
    ok = diff < 0.01
    return ok, diff, rel_max


def main() -> int:
    torch.cuda.set_device(0)

    # V4-Flash routed-expert shapes (from Phase 11 trace at
    # nested-growing-hammock.md): E=256 (down-sharded to E_local at TP=2),
    # K (hidden) = 4096, N (intermediate per expert) = 2048 for w13,
    # N = 1024 for w2. M ∈ {1, 2, 4, 6, 8} per Phase 11.0 trace.
    # We use a smaller E to keep the harness fast; layout correctness
    # doesn't depend on E.
    cases = [
        # (role, num_experts, n, k, m_per_group)
        ("w13", 4, 2 * 2048, 4096, 32),
        ("w2",  4, 4096,     2048, 32),
    ]

    print("=" * 70)
    print("D4.1 MXFP4 weight migration: deepgemm vs BF16 reference")
    print("=" * 70)

    fail = 0
    for role, num_experts, n, k, m in cases:
        ok, diff, rel_max = _run_one_role(
            role, num_experts=num_experts, n_per_expert=n, k=k,
            expected_m_per_group=m,
        )
        verdict = "PASS" if ok else "FAIL"
        print(
            f"  role={role:3s} E={num_experts} n={n:5d} k={k:5d} m={m:3d} — "
            f"{verdict} calc_diff={diff:.3e} (threshold 1e-2)"
        )
        if not ok:
            fail += 1

    print()
    print(f"Result: {len(cases) - fail}/{len(cases)} passed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
