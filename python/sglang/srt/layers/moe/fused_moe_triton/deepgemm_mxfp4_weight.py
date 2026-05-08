"""Convert V4-Flash MXFP4 routed-expert weights to deepgemm-sm120's
expected B-operand layout for `m_grouped_fp8_fp4_gemm_nt_contiguous`.

V4-Flash checkpoint loads MXFP4 weights as `torch.uint8` (packed
nibbles, 2 elements per byte) with companion `torch.uint8` UE8M0
scale tensors. Deepgemm's grouped FP8×FP4 GEMM expects the B operand
as a tuple `(packed_nibbles_int8, scales_fp32)` with shapes:

- packed_nibbles_int8: `[num_groups, n, k // 2]` int8, K-major.
- scales_fp32:        `[num_groups, n, ceil_div(k, gran_k)]` fp32.

The scale tensor is decoded from UE8M0 bytes via
`scale_fp32 = 2.0**(byte - 127.0)`. Deepgemm internally re-packs
to INT32 UE8M0 via `transform_sf_into_required_layout` when
`disable_ue8m0_cast=False`, so the helper does NOT need to pre-pack.

Two roles are handled separately because their shapes differ:

- w13 (gate + up packed together): raw shape
  `[E, 2 * intermediate, hidden // 2]`.
- w2 (down projection): raw shape
  `[E, hidden, intermediate // 2]`.
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch


def _prepare_deepgemm_mxfp4_weight(
    raw_weight: torch.Tensor,
    raw_scale: torch.Tensor,
    *,
    role: Literal["w13", "w2"],
    gran_k: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert raw V4-Flash MXFP4 (uint8 packed nibbles + uint8 UE8M0
    scales) into deepgemm B-operand format.

    Args:
        raw_weight: torch.uint8, shape `[E, n, k // 2]` for K-major
            packed FP4 nibbles. For `role="w13"` n=2*intermediate,
            k=hidden. For `role="w2"` n=hidden, k=intermediate.
        raw_scale: torch.uint8, shape `[E, n, k // gran_k]` UE8M0
            exponent bytes.
        role: which weight role, used only for assertion shape sanity.
        gran_k: K-direction group size for UE8M0 scales (default 32).

    Returns:
        weight_int8: deepgemm-compatible packed FP4 nibbles, dtype
            `torch.int8`, same shape as `raw_weight` (uint8 reinterpreted
            as int8 — same bytes, just typed for the deepgemm contract).
        scales_fp32: decoded scales `2.0**(byte - 127.0)`, dtype
            `torch.float32`, same shape as `raw_scale`.
    """
    assert raw_weight.dtype == torch.uint8, (
        f"expected uint8 weight, got {raw_weight.dtype}"
    )
    assert raw_scale.dtype == torch.uint8, (
        f"expected uint8 UE8M0 scale, got {raw_scale.dtype}"
    )
    assert raw_weight.dim() == 3 and raw_scale.dim() == 3, (
        f"expected 3-D weight + scale, got weight.dim()={raw_weight.dim()} "
        f"scale.dim()={raw_scale.dim()}"
    )
    assert raw_weight.shape[0] == raw_scale.shape[0], (
        f"E mismatch: weight.shape[0]={raw_weight.shape[0]} "
        f"scale.shape[0]={raw_scale.shape[0]}"
    )
    assert raw_weight.shape[1] == raw_scale.shape[1], (
        f"n mismatch: weight.shape[1]={raw_weight.shape[1]} "
        f"scale.shape[1]={raw_scale.shape[1]}"
    )
    # k_packed = k // 2 (two FP4 nibbles per byte). The scale tensor's
    # last dim should be k // gran_k, so:
    #   2 * weight.shape[2] / gran_k == scale.shape[2].
    k_packed = raw_weight.shape[2]
    expected_scale_k = (2 * k_packed) // gran_k
    assert raw_scale.shape[2] == expected_scale_k, (
        f"k mismatch: scale.shape[2]={raw_scale.shape[2]} "
        f"expected {expected_scale_k} from weight.shape[2]={k_packed} "
        f"with gran_k={gran_k}"
    )
    _ = role  # informational; shapes are validated above

    # uint8 -> int8: same bytes, just a different dtype tag for deepgemm's
    # "packed FP4" contract (matches `tests/generators.py:260` which uses
    # `torch.int8` for the packed FP4 weight tensor).
    weight_int8 = raw_weight.view(torch.int8)

    # UE8M0 byte -> FP32 scale via 2^(byte - 127), then pre-pack to
    # deepgemm's INT32 TMA-aligned layout. The packed output is ~1/4
    # the FP32 size which matters at V4-Flash scales (~400MB → ~100MB
    # per rank for the full E×N×K/32 scale tensor).
    scales_fp32 = torch.pow(2.0, raw_scale.to(torch.float32) - 127.0)
    from deep_gemm import transform_sf_into_required_layout
    num_groups, n, _ = raw_weight.shape
    k = 2 * raw_weight.shape[2]  # nibbles → elements
    scales_packed = transform_sf_into_required_layout(
        scales_fp32, mn=n, k=k, recipe=(1, gran_k), num_groups=num_groups,
    )
    del scales_fp32  # release the transient FP32 alloc

    return weight_int8, scales_packed
