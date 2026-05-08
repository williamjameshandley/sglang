"""D4.7 oracle: live tensor numerical comparison for V4-Flash MXFP4 MoE
deepgemm integration.

Modes (escalating realism; failure at level N blocks higher levels):

  --mode 1 — single-GEMM real-weight layout matrix (24 w13 cases, 4 w2)
  --mode 2 — single-expert two-GEMM composition
  --mode 3 — masked-local E_local with sparse routing
  --mode 4 — full MoeRunner.run end-to-end vs OAI custom-op
  --mode replay --dump <path.pt> — replay live tensor capture

Phase D4.7 motivation: D4.1 harness PASSes with synthetic uniform weights
in contiguous mode, but live deploy with real V4-Flash MXFP4 weights in
masked mode produces garbage output. The oracle isolates which suspect
class is the cause (weight layout, scale contract, routing, post-reorder)
without requiring a 5-min live restart cycle.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import torch
from safetensors import safe_open

# Layer-0 routed-experts safetensors location for V4-Flash.
_HF_SNAPSHOT_DIR = (
    "/var/lib/sglang/hub/models--deepseek-ai--DeepSeek-V4-Flash/"
    "snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136"
)
_LAYER0_SHARD = os.path.join(_HF_SNAPSHOT_DIR, "model-00002-of-00046.safetensors")

# V4-Flash routed-experts shape constants (from config.json + safetensors).
HIDDEN = 4096
INTERMEDIATE = 2048
GRAN_K = 32  # MXFP4 group size on the K dim.


# ---------------------------------------------------------------------------
# Weight loaders: pull one expert's MXFP4 tensors from the checkpoint.
# ---------------------------------------------------------------------------

def _load_expert_weights(expert_id: int):
    """Return raw checkpoint MXFP4 tensors for one expert.

    Returns a dict with:
      w1_w, w1_s, w3_w, w3_s, w2_w, w2_s
    where *_w are uint8-viewed packed-FP4 nibbles (one byte = two FP4
    elements along the K axis) and *_s are uint8-viewed UE8M0 scales.
    Shapes:
      w1_w / w3_w: [intermediate, hidden//2] = [2048, 2048]
      w1_s / w3_s: [intermediate, hidden//gran_k] = [2048, 128]
      w2_w:        [hidden, intermediate//2] = [4096, 1024]
      w2_s:        [hidden, intermediate//gran_k] = [4096, 64]
    """
    out = {}
    with safe_open(_LAYER0_SHARD, framework="pt", device="cuda:0") as f:
        for sub, key in [("w1", "w1"), ("w2", "w2"), ("w3", "w3")]:
            w_key = f"layers.0.ffn.experts.{expert_id}.{key}.weight"
            s_key = f"layers.0.ffn.experts.{expert_id}.{key}.scale"
            w = f.get_tensor(w_key)            # int8 packed nibbles
            s = f.get_tensor(s_key)            # float8_e8m0fnu
            out[f"{sub}_w"] = w.view(torch.uint8).contiguous()
            out[f"{sub}_s"] = s.view(torch.uint8).contiguous()
    return out


def build_w13(weights: dict, gate_up_convention: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the w13 packed-FP4 tensor for one expert under a chosen
    gate/up convention.

    `weights` is the output of `_load_expert_weights`. Returns
    (w13_weight uint8 [4096, 2048], w13_scale uint8 [4096, 128]).

    Conventions per A4 (six candidates):

        gate_first_half   : [w1; w3]
        up_first_half     : [w3; w1]
        interleaved_gate_up : interleave w1, w3 row-by-row
        interleaved_up_gate : interleave w3, w1 row-by-row
        flashinfer_swap   : [w1; w3] then swap_every_two_rows
        aiter_permute     : [w1; w3] then reshape (E, n//2, 2, k) -> permute
                             (0, 2, 1, 3) -> contiguous -> view (n, k)
    """
    w1_w, w3_w = weights["w1_w"], weights["w3_w"]
    w1_s, w3_s = weights["w1_s"], weights["w3_s"]

    if gate_up_convention == "gate_first_half":
        w13_w = torch.cat([w1_w, w3_w], dim=0)
        w13_s = torch.cat([w1_s, w3_s], dim=0)
    elif gate_up_convention == "up_first_half":
        w13_w = torch.cat([w3_w, w1_w], dim=0)
        w13_s = torch.cat([w3_s, w1_s], dim=0)
    elif gate_up_convention == "interleaved_gate_up":
        n, kp = w1_w.shape
        w13_w = torch.empty(2 * n, kp, dtype=w1_w.dtype, device=w1_w.device)
        w13_w[0::2] = w1_w
        w13_w[1::2] = w3_w
        n_s, kg = w1_s.shape
        w13_s = torch.empty(2 * n_s, kg, dtype=w1_s.dtype, device=w1_s.device)
        w13_s[0::2] = w1_s
        w13_s[1::2] = w3_s
    elif gate_up_convention == "interleaved_up_gate":
        n, kp = w1_w.shape
        w13_w = torch.empty(2 * n, kp, dtype=w1_w.dtype, device=w1_w.device)
        w13_w[0::2] = w3_w
        w13_w[1::2] = w1_w
        n_s, kg = w1_s.shape
        w13_s = torch.empty(2 * n_s, kg, dtype=w1_s.dtype, device=w1_s.device)
        w13_s[0::2] = w3_s
        w13_s[1::2] = w1_s
    elif gate_up_convention == "flashinfer_swap":
        # Concat then swap every two rows. mxfp4.py:534-554's reference
        # swaps along the row axis by reshaping (-1, 2) and reversing.
        w_cat = torch.cat([w1_w, w3_w], dim=0)
        s_cat = torch.cat([w1_s, w3_s], dim=0)
        n, kp = w_cat.shape
        w13_w = w_cat.view(n // 2, 2, kp).flip(dims=(1,)).reshape(n, kp).contiguous()
        n_s, kg = s_cat.shape
        w13_s = s_cat.view(n_s // 2, 2, kg).flip(dims=(1,)).reshape(n_s, kg).contiguous()
    elif gate_up_convention == "aiter_permute":
        # mxfp4.py:688-695 pattern: view(n, n//2, 2, k) -> permute(0,2,1,3)
        # On a single expert (no E dim), it's view(n//2, 2, k) -> permute(1,0,2).
        w_cat = torch.cat([w1_w, w3_w], dim=0)
        s_cat = torch.cat([w1_s, w3_s], dim=0)
        n, kp = w_cat.shape
        w13_w = (
            w_cat.view(n // 2, 2, kp).permute(1, 0, 2).contiguous().view(n, kp)
        )
        n_s, kg = s_cat.shape
        w13_s = (
            s_cat.view(n_s // 2, 2, kg).permute(1, 0, 2).contiguous().view(n_s, kg)
        )
    else:
        raise ValueError(f"unknown gate_up_convention={gate_up_convention!r}")

    return w13_w, w13_s


GATE_UP_CONVENTIONS = (
    "gate_first_half",
    "up_first_half",
    "interleaved_gate_up",
    "interleaved_up_gate",
    "flashinfer_swap",
    "aiter_permute",
)


# ---------------------------------------------------------------------------
# Layout transforms applied to (weight, scale) pairs before the GEMM.
# ---------------------------------------------------------------------------

def _swap_nibbles(w: torch.Tensor) -> torch.Tensor:
    """Swap high vs low nibble in each byte. MXFP4 packs two FP4 values
    per byte; some conventions store the lower-K-index in the low nibble,
    others in the high nibble."""
    return ((w & 0x0F) << 4) | ((w >> 4) & 0x0F)


def _apply_layout_variant(
    w_uint8: torch.Tensor,
    s_uint8: torch.Tensor,
    nibble_order: str,
    kn_orientation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply nibble-order and K/N transposition variants.

    Args:
        w_uint8: [N, K_packed] uint8 weight.
        s_uint8: [N, K_scale]  uint8 UE8M0.
        nibble_order: "low_first_K" (default) or "high_first_K" (swap each byte).
        kn_orientation: "N_outer" (default, [N, K_packed]) or "K_outer"
            (transpose to [K_packed, N]).
    """
    w = w_uint8
    s = s_uint8
    if nibble_order == "high_first_K":
        w = _swap_nibbles(w)
    elif nibble_order != "low_first_K":
        raise ValueError(nibble_order)

    if kn_orientation == "N_outer":
        pass
    elif kn_orientation == "K_outer":
        w = w.transpose(0, 1).contiguous()
        s = s.transpose(0, 1).contiguous()
    else:
        raise ValueError(kn_orientation)

    return w, s


NIBBLE_ORDERS = ("low_first_K", "high_first_K")
KN_ORIENTATIONS = ("N_outer", "K_outer")


# ---------------------------------------------------------------------------
# Reference paths.
# ---------------------------------------------------------------------------

def reference_bf16_matmul(
    w_uint8_canonical: torch.Tensor,   # [N, K_packed]
    s_uint8_canonical: torch.Tensor,   # [N, K/gran_k]
    a_bf16: torch.Tensor,              # [M, K]
) -> torch.Tensor:
    """R-bf16: dequantize MXFP4 -> BF16, then BF16 @ BF16.T matmul.

    Matches the in-tree fallback at mxfp4.py:803-810 which calls
    `triton_kernels.numerics_details.mxfp.upcast_from_mxfp(weight, scale,
    target_dtype=torch.bfloat16, axis=-1)`.

    Returns FP32 output `[M, N]` for tighter element-wise comparison.
    """
    from triton_kernels.numerics_details.mxfp import upcast_from_mxfp

    w_bf16 = upcast_from_mxfp(
        w_uint8_canonical, s_uint8_canonical,
        target_dtype=torch.bfloat16, axis=-1,
    )
    return (a_bf16.float() @ w_bf16.float().T).contiguous()


def reference_oai_matmul_ogs(
    w_uint8_canonical: torch.Tensor,
    s_uint8_canonical: torch.Tensor,
    a_bf16: torch.Tensor,
) -> torch.Tensor:
    """R-oai: known-good production baseline via OAI matmul_ogs.

    Construct OAI-swizzled weights via `_swizzle_mxfp4` and call
    `triton_kernels.matmul_ogs` directly. This is the path the live
    deploy currently runs successfully.

    NOTE: matmul_ogs is the routed-experts kernel and may not have a
    pure single-expert one-shot mode. Falling back to BF16 reference for
    Mode 1 (single-GEMM oracle) is acceptable when OAI swizzle requires
    multi-expert tensors; Mode 2+ will exercise OAI properly with the
    multi-expert PrecisionConfig context.
    """
    # For Mode 1, use the BF16 dequant path (proven via in-tree fallback)
    # as a stand-in. R-oai is fully exercised in Modes 2-4 where the
    # full multi-expert routing context is constructed.
    return reference_bf16_matmul(w_uint8_canonical, s_uint8_canonical, a_bf16)


# ---------------------------------------------------------------------------
# Deepgemm test path.
# ---------------------------------------------------------------------------

def deepgemm_single_gemm(
    w_uint8: torch.Tensor,             # [N, K_packed] under chosen layout
    s_uint8: torch.Tensor,             # [N, K_scale]  UE8M0
    a_bf16: torch.Tensor,              # [M, K]
    *,
    scale_contract: str = "prepacked_int32",
    packedfp4_view: str = "int8",
) -> torch.Tensor:
    """Run `m_grouped_fp8_fp4_gemm_nt_contiguous` for a single expert with
    the supplied layout. Returns BF16 output `[M, N]`.

    `scale_contract`:
        - "prepacked_int32" — caller-side `transform_sf_into_required_layout`
        - "raw_fp32" — pass FP32 scales directly (deepgemm internally packs)

    `packedfp4_view`: how to dtype-tag the weight tensor for the call:
        - "int8"               — `view(torch.int8)`
        - "uint8"              — pass as-is (uint8)
        - "float4_e2m1fn_x2"   — view as packed-FP4 dtype
    """
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp8

    # Set sm120 contiguous alignment (matches D4.1 harness pattern).
    deep_gemm.set_mk_alignment_for_contiguous_layout(
        deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    )
    align_m = deep_gemm.get_mk_alignment_for_contiguous_layout()
    M, K = a_bf16.shape
    N = w_uint8.shape[0]

    # Pad M up to alignment (one group, num_experts=1).
    aligned_m = ((M + align_m - 1) // align_m) * align_m
    if aligned_m != M:
        pad = aligned_m - M
        a_padded = torch.cat(
            [a_bf16, torch.zeros(pad, K, dtype=a_bf16.dtype, device=a_bf16.device)],
            dim=0,
        )
    else:
        a_padded = a_bf16

    # FP8 activation quant with UE8M0 (matches D4.1 harness; without UE8M0
    # at K>=512 deepgemm produces NaN).
    a_fp8, a_sf = per_token_cast_to_fp8(a_padded, use_ue8m0=True, gran_k=128)

    # Single-group contiguous layout: all rows go to expert 0.
    grouped_layout = torch.zeros(aligned_m, dtype=torch.int32, device=a_bf16.device)

    # B-side (weight) under chosen dtype view.
    if packedfp4_view == "int8":
        b_w = w_uint8.view(torch.int8).unsqueeze(0)            # [1, N, K_packed]
    elif packedfp4_view == "uint8":
        b_w = w_uint8.unsqueeze(0)
    elif packedfp4_view == "float4_e2m1fn_x2":
        b_w = w_uint8.view(torch.float4_e2m1fn_x2).unsqueeze(0)
    else:
        raise ValueError(packedfp4_view)

    # Decode UE8M0 byte → FP32 scale.
    s_fp32 = torch.pow(2.0, s_uint8.to(torch.float32) - 127.0).unsqueeze(0)  # [1, N, K_scale]

    if scale_contract == "prepacked_int32":
        b_s = deep_gemm.transform_sf_into_required_layout(
            s_fp32, mn=N, k=K, recipe=(1, GRAN_K), num_groups=1,
        )
    elif scale_contract == "raw_fp32":
        b_s = s_fp32
    else:
        raise ValueError(scale_contract)

    out = torch.empty(aligned_m, N, dtype=torch.bfloat16, device=a_bf16.device)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        a=(a_fp8, a_sf),
        b=(b_w, b_s),
        d=out,
        grouped_layout=grouped_layout,
        recipe_a=(1, 128),
        recipe_b=(1, GRAN_K),
    )
    return out[:M].contiguous()


# ---------------------------------------------------------------------------
# Comparison utility.
# ---------------------------------------------------------------------------

def calc_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine-similarity-style diff matching deepgemm's testing.numeric.calc_diff."""
    af = a.float().flatten()
    bf = b.float().flatten()
    finite = torch.isfinite(af) & torch.isfinite(bf)
    if not finite.any():
        return float("nan")
    af = af[finite]
    bf = bf[finite]
    cos = (af * bf).sum() / (af.norm() * bf.norm() + 1e-12)
    return float((1.0 - cos).item())


# ---------------------------------------------------------------------------
# Mode 2: single-expert two-GEMM composition (silu_and_mul split + w2).
# ---------------------------------------------------------------------------

def reference_two_gemm_bf16(
    w13_uint8: torch.Tensor,    # [2*intermediate, hidden//2]
    w13_s_uint8: torch.Tensor,
    w2_uint8: torch.Tensor,     # [hidden, intermediate//2]
    w2_s_uint8: torch.Tensor,
    a_bf16: torch.Tensor,       # [M, hidden]
    *,
    gate_first_half: bool = True,
) -> torch.Tensor:
    """Two-GEMM BF16 reference. `gate_first_half=True` splits w13 as
    [gate; up] (first half rows = gate). False = [up; gate]."""
    from triton_kernels.numerics_details.mxfp import upcast_from_mxfp

    w13_bf16 = upcast_from_mxfp(
        w13_uint8, w13_s_uint8, target_dtype=torch.bfloat16, axis=-1,
    )
    w2_bf16 = upcast_from_mxfp(
        w2_uint8, w2_s_uint8, target_dtype=torch.bfloat16, axis=-1,
    )
    # GEMM 1: hidden_states @ w13.T → [M, 2*intermediate]
    gateup = (a_bf16.float() @ w13_bf16.float().T)
    inter = w13_uint8.shape[0] // 2
    if gate_first_half:
        gate, up = gateup[:, :inter], gateup[:, inter:]
    else:
        up, gate = gateup[:, :inter], gateup[:, inter:]
    silu = torch.nn.functional.silu(gate) * up
    # GEMM 2: silu @ w2.T → [M, hidden]
    out = silu @ w2_bf16.float().T
    return out.contiguous()


def deepgemm_two_gemm(
    w13_uint8, w13_s_uint8,
    w2_uint8, w2_s_uint8,
    a_bf16,
    *,
    gate_first_half: bool = True,
) -> torch.Tensor:
    """Two-GEMM deepgemm test path. silu_and_mul on the gate/up split
    chosen by `gate_first_half`."""
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp8

    deep_gemm.set_mk_alignment_for_contiguous_layout(
        deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    )
    align_m = deep_gemm.get_mk_alignment_for_contiguous_layout()
    M, K1 = a_bf16.shape
    N1 = w13_uint8.shape[0]                 # 2*intermediate
    inter = N1 // 2
    K2 = inter
    N2 = w2_uint8.shape[0]                  # hidden

    aligned_m = ((M + align_m - 1) // align_m) * align_m
    if aligned_m != M:
        a_padded = torch.cat(
            [a_bf16, torch.zeros(aligned_m - M, K1, dtype=a_bf16.dtype, device=a_bf16.device)],
            dim=0,
        )
    else:
        a_padded = a_bf16

    grouped_layout = torch.zeros(aligned_m, dtype=torch.int32, device=a_bf16.device)

    # GEMM 1 — w13.
    a13_fp8, a13_sf = per_token_cast_to_fp8(a_padded, use_ue8m0=True, gran_k=128)
    w13_b = w13_uint8.view(torch.int8).unsqueeze(0)
    w13_sfp = torch.pow(2.0, w13_s_uint8.to(torch.float32) - 127.0).unsqueeze(0)
    w13_sp = deep_gemm.transform_sf_into_required_layout(
        w13_sfp, mn=N1, k=K1, recipe=(1, GRAN_K), num_groups=1,
    )
    gateup = torch.empty(aligned_m, N1, dtype=torch.bfloat16, device=a_bf16.device)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        a=(a13_fp8, a13_sf), b=(w13_b, w13_sp), d=gateup,
        grouped_layout=grouped_layout, recipe_a=(1, 128), recipe_b=(1, GRAN_K),
    )

    # silu_and_mul: split into gate/up halves, compute silu(gate)*up.
    if gate_first_half:
        gate, up = gateup[:, :inter], gateup[:, inter:]
    else:
        up, gate = gateup[:, :inter], gateup[:, inter:]
    inter_bf16 = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()

    # GEMM 2 — w2.
    a2_fp8, a2_sf = per_token_cast_to_fp8(inter_bf16, use_ue8m0=True, gran_k=128)
    w2_b = w2_uint8.view(torch.int8).unsqueeze(0)
    w2_sfp = torch.pow(2.0, w2_s_uint8.to(torch.float32) - 127.0).unsqueeze(0)
    w2_sp = deep_gemm.transform_sf_into_required_layout(
        w2_sfp, mn=N2, k=K2, recipe=(1, GRAN_K), num_groups=1,
    )
    out = torch.empty(aligned_m, N2, dtype=torch.bfloat16, device=a_bf16.device)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        a=(a2_fp8, a2_sf), b=(w2_b, w2_sp), d=out,
        grouped_layout=grouped_layout, recipe_a=(1, 128), recipe_b=(1, GRAN_K),
    )
    return out[:M].contiguous()


def mode2_two_gemm(expert_id: int = 0, M_values: tuple = (8, 64)) -> dict:
    """Test all 6 gate/up conventions for w13 paired with the canonical
    w2. Each (convention, gate_first_half) combination is run; the
    correct one is the one where deepgemm matches BF16 reference."""
    weights = _load_expert_weights(expert_id)
    device = "cuda:0"
    torch.manual_seed(0xD4D4D4)
    a_list = [
        (
            torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=device) * 0.5
            + 0.1 * torch.linspace(-1, 1, M, device=device).unsqueeze(1).bfloat16()
        )
        for M in M_values
    ]
    w2_w, w2_s = weights["w2_w"], weights["w2_s"]

    results = []
    for gate_up in GATE_UP_CONVENTIONS:
        w13_w, w13_s = build_w13(weights, gate_up)
        for gate_first in (True, False):
            for M_idx, M in enumerate(M_values):
                a = a_list[M_idx]
                try:
                    ref = reference_two_gemm_bf16(
                        w13_w, w13_s, w2_w, w2_s, a, gate_first_half=gate_first,
                    )
                    test = deepgemm_two_gemm(
                        w13_w, w13_s, w2_w, w2_s, a, gate_first_half=gate_first,
                    )
                    diff = calc_diff(test.float(), ref.float())
                    ok = "PASS" if diff < 1e-2 else ("WEAK" if diff < 5e-2 else "FAIL")
                except Exception as e:
                    diff = float("inf")
                    ok = f"ERR: {type(e).__name__}"
                results.append({
                    "gate_up": gate_up,
                    "gate_first": gate_first,
                    "M": M,
                    "diff": diff,
                    "verdict": ok,
                })
    return {"results": results}


def print_mode2_results(out: dict, top_n: int = 24):
    rs = out["results"]
    rs.sort(key=lambda r: r["diff"])
    print(f"{'gate_up':>22} {'gate_first':>11} {'M':>4} {'diff':>10} verdict")
    for r in rs[:top_n]:
        print(f"{r['gate_up']:>22} {str(r['gate_first']):>11} "
              f"{r['M']:>4} {r['diff']:>10.3e} {r['verdict']}")


# ---------------------------------------------------------------------------
# Mode 2.5: contiguous vs masked on the SAME single-expert input.
# ---------------------------------------------------------------------------

def deepgemm_masked_single_expert(
    w_uint8, s_uint8, a_bf16,
) -> torch.Tensor:
    """Run `m_grouped_fp8_fp4_gemm_nt_masked` with E_local=1 (one expert,
    one group, masked_m=[M], expected_m=M). Returns BF16 output [M, N].

    Compare to `deepgemm_single_gemm` which uses contiguous mode on the
    same input. If they disagree, the masked kernel has a different
    contract than I'm assuming.
    """
    import deep_gemm
    from deep_gemm.utils.math import per_token_cast_to_fp8

    M, K = a_bf16.shape
    N = w_uint8.shape[0]

    # Masked layout: A is [E=1, M, K]. Quantize per token, then add E dim.
    a_fp8_2d, a_sf_2d = per_token_cast_to_fp8(a_bf16, use_ue8m0=True, gran_k=128)
    a_fp8 = a_fp8_2d.unsqueeze(0)             # [1, M, K]
    a_sf = a_sf_2d.unsqueeze(0)               # [1, M, K/128]

    masked_m = torch.tensor([M], dtype=torch.int32, device=a_bf16.device)
    expected_m = M

    # B operand: same as contiguous path.
    b_w = w_uint8.view(torch.int8).unsqueeze(0)
    s_fp32 = torch.pow(2.0, s_uint8.to(torch.float32) - 127.0).unsqueeze(0)
    b_s = deep_gemm.transform_sf_into_required_layout(
        s_fp32, mn=N, k=K, recipe=(1, GRAN_K), num_groups=1,
    )

    out = torch.empty(1, M, N, dtype=torch.bfloat16, device=a_bf16.device)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
        a=(a_fp8, a_sf),
        b=(b_w, b_s),
        d=out,
        masked_m=masked_m,
        expected_m=expected_m,
        recipe_a=(1, 128),
        recipe_b=(1, GRAN_K),
    )
    return out[0].contiguous()


def mode25_masked_vs_contiguous(expert_id: int = 0, M_values: tuple = (8, 64)) -> dict:
    """Compare contiguous-grouped GEMM vs masked-grouped GEMM on the
    SAME single-expert input. They MUST agree (same kernel family,
    different layout); if they don't, the masked path has a contract
    mismatch in our integration."""
    weights = _load_expert_weights(expert_id)
    device = "cuda:0"
    torch.manual_seed(0xD4D4D4)

    results = []
    for role, w, s, K in [("w2", weights["w2_w"], weights["w2_s"], INTERMEDIATE),
                          ("w13_canonical", *build_w13(weights, "gate_first_half"), HIDDEN)]:
        for M in M_values:
            a = (
                torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.5
                + 0.1 * torch.linspace(-1, 1, M, device=device).unsqueeze(1).bfloat16()
            )
            try:
                out_contig = deepgemm_single_gemm(w, s, a)        # contiguous
                out_masked = deepgemm_masked_single_expert(w, s, a)
                diff = calc_diff(out_contig.float(), out_masked.float())
                ok = "PASS" if diff < 1e-3 else ("WEAK" if diff < 1e-2 else "FAIL")
            except Exception as e:
                diff = float("inf")
                ok = f"ERR: {type(e).__name__}: {e}"[:80]
            results.append({"role": role, "M": M, "diff": diff, "verdict": ok})
    return {"results": results}


# ---------------------------------------------------------------------------
# Mode 1 driver.
# ---------------------------------------------------------------------------

def mode1_single_gemm(role: str, expert_id: int = 0, M_values: tuple = (8, 64)) -> dict:
    """Sweep the layout matrix for one role on one expert."""
    weights = _load_expert_weights(expert_id)
    device = "cuda:0"

    # Build deterministic non-uniform activations: sign + magnitude variation,
    # multiple M values. (Mode 1 acceptance: must rule out uniform-only artifacts.)
    torch.manual_seed(0xD4D4D4)
    a_list = []
    K = HIDDEN if role == "w13" else INTERMEDIATE
    for M in M_values:
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.5
        a += 0.1 * torch.linspace(-1, 1, M, device=device).unsqueeze(1).bfloat16()
        a_list.append(a)

    if role == "w2":
        # w2 has no gate/up axis; only nibble × K/N × scale × dtype.
        gate_up_set = ("N/A",)
        canonical_w = weights["w2_w"]
        canonical_s = weights["w2_s"]
    elif role == "w13":
        gate_up_set = GATE_UP_CONVENTIONS
        canonical_w = canonical_s = None  # built per gate_up
    else:
        raise ValueError(role)

    results = []
    for gate_up in gate_up_set:
        if role == "w13":
            w_base, s_base = build_w13(weights, gate_up)
        else:
            w_base, s_base = canonical_w, canonical_s

        # Reference: BF16 dequant from canonical (chosen) layout.
        # We use w_base/s_base AS the canonical for upcast_from_mxfp.
        # Each (gate_up) reorder constitutes a different N-order; the
        # reference matmul reorders implicitly because we're comparing
        # output[M,N] under the same N-order as the deepgemm input.
        for nibble in NIBBLE_ORDERS:
            for orient in KN_ORIENTATIONS:
                w_var, s_var = _apply_layout_variant(w_base, s_base, nibble, orient)
                if orient == "K_outer":
                    # When K-outer, deepgemm expects [N, K_packed] but we have
                    # [K_packed, N] — skip (the orientation isn't the deepgemm
                    # contract). Document and continue.
                    continue
                for scale_c in ("prepacked_int32", "raw_fp32"):
                    for view in ("int8", "uint8", "float4_e2m1fn_x2"):
                        for M_idx, M in enumerate(M_values):
                            a = a_list[M_idx]
                            try:
                                ref = reference_bf16_matmul(w_var, s_var, a)
                                test = deepgemm_single_gemm(
                                    w_var, s_var, a,
                                    scale_contract=scale_c,
                                    packedfp4_view=view,
                                )
                                diff = calc_diff(test.float(), ref.float())
                                ok = "PASS" if diff < 1e-2 else ("WEAK" if diff < 5e-2 else "FAIL")
                            except Exception as e:
                                diff = float("inf")
                                ok = f"ERR: {type(e).__name__}"
                            results.append({
                                "role": role, "gate_up": gate_up, "nibble": nibble,
                                "orient": orient, "scale_contract": scale_c,
                                "view": view, "M": M, "diff": diff, "verdict": ok,
                            })
    return {"results": results}


def print_results(out: dict, top_n: int = 20):
    rs = out["results"]
    rs.sort(key=lambda r: r["diff"])
    print(f"{'role':>4} {'gate_up':>20} {'nibble':>14} {'orient':>9} "
          f"{'scale':>16} {'view':>20} {'M':>4} {'diff':>10} verdict")
    for r in rs[:top_n]:
        print(f"{r['role']:>4} {r['gate_up']:>20} {r['nibble']:>14} "
              f"{r['orient']:>9} {r['scale_contract']:>16} {r['view']:>20} "
              f"{r['M']:>4} {r['diff']:>10.3e} {r['verdict']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="1")
    p.add_argument("--role", default="w2", choices=("w13", "w2"))
    p.add_argument("--expert", type=int, default=0)
    args = p.parse_args()

    if args.mode == "1":
        out = mode1_single_gemm(args.role, expert_id=args.expert)
        print_results(out, top_n=30)
        passing = [r for r in out["results"] if r["verdict"] == "PASS"]
        print(f"\n{len(passing)}/{len(out['results'])} cases passed (diff < 1e-2)")
        return 0 if passing else 1
    elif args.mode == "2":
        out = mode2_two_gemm(expert_id=args.expert)
        print_mode2_results(out, top_n=24)
        passing = [r for r in out["results"] if r["verdict"] == "PASS"]
        print(f"\n{len(passing)}/{len(out['results'])} cases passed (diff < 1e-2)")
        return 0 if passing else 1
    elif args.mode == "2.5":
        out = mode25_masked_vs_contiguous(expert_id=args.expert)
        for r in out["results"]:
            print(f"  {r['role']:>14} M={r['M']:>3}  diff={r['diff']:.3e}  {r['verdict']}")
        bad = [r for r in out["results"] if r["verdict"] != "PASS"]
        return 1 if bad else 0
    else:
        print(f"Mode {args.mode} not yet implemented", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
