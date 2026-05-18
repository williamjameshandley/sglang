"""Offline replay-bisect analyzer for the D4.7.5b staged live dump.

Consumes three dump files produced by a single live run with
SGLANG_OPT_DUMP_DEEPGEMM_MOE=1:

  /tmp/deepgemm_stages_<pid>.pt    (staged tensors from moe_runner/deep_gemm.py)
  /tmp/deepgemm_dump_<pid>.pt      (input-side dump from mxfp4.py.apply)
  /tmp/deepgemm_dumpout_<pid>.pt   (final post-runner output from mxfp4.py.apply)

For each stage, computes the least-squares scalar α that best fits the
live tensor against a BF16 reference reconstructed from the raw
checkpoint weights and the captured upstream tensors. The first stage
where |α - 1.94| < tol localises the bug.

Stages:
  post_gemm1               : gateup_output_active vs BF16 reference
  post_activation          : dequant(down_input) vs silu(gate)*up
  post_gemm2               : down_output_active vs BF16 reference
  post_reorder_before_rsf  : output vs manual src2dst+topk_weights combine
  post_reorder_after_rsf   : output vs before_rsf * routed_scaling_factor

Usage:
  python _replay_deepgemm_stages.py \\
      --stages /tmp/deepgemm_stages_<pid>.pt \\
      --inputs /tmp/deepgemm_dump_<pid>.pt \\
      --output /tmp/deepgemm_dumpout_<pid>.pt \\
      [--rank 0] [--tp 2]
"""

from __future__ import annotations

import argparse
import sys

import torch


# Reuse the oracle's checkpoint loader / rank-sharder.
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _test_deepgemm_moe_oracle import (  # noqa: E402
    _load_expert_weights, _rank_shard_components, build_w13,
    GATE_UP_CONVENTIONS,
)


def _unpack_ue8m0_int32_to_fp32(scale_i32: torch.Tensor) -> torch.Tensor:
    """Unpack DeepGEMM's packed UE8M0 INT32 scale layout to FP32.

    Each int32 holds 4 UE8M0 exponent bytes. The packed tensor has logical
    shape `[..., K_packed]` where the unpacked logical shape is
    `[..., K_packed * 4]`. Each byte b decodes as `2.0**(b - 127.0)`.

    The live tensor may have come through
    `_cast_to_e8m0_with_rounding_up`'s
    `transpose(1,2).contiguous().transpose(1,2)` step, which leaves the
    layout column-major. `.contiguous()` here normalises before reading
    bytes.
    """
    s = scale_i32.contiguous()
    b = s.view(torch.uint8).reshape(*s.shape[:-1], s.shape[-1] * 4)
    return torch.pow(2.0, b.to(torch.float32) - 127.0)


def alpha_metrics(test: torch.Tensor, ref: torch.Tensor) -> dict:
    t = test.float().flatten()
    r = ref.float().flatten()
    finite = torch.isfinite(t) & torch.isfinite(r)
    t = t[finite]
    r = r[finite]
    if t.numel() == 0:
        return {"alpha": float("nan"), "rel_l2": float("nan"),
                "rel_resid_after_alpha": float("nan"), "n": 0}
    rr = (r * r).sum().clamp_min(1e-12)
    alpha = (t * r).sum() / rr
    rel_l2 = (t - r).norm() / r.norm().clamp_min(1e-12)
    resid = t - alpha * r
    rel_resid = resid.norm() / t.norm().clamp_min(1e-12)
    return {"alpha": float(alpha), "rel_l2": float(rel_l2),
            "rel_resid_after_alpha": float(rel_resid),
            "n": int(t.numel())}


def bf16_dequant_w(w_uint8: torch.Tensor, s_uint8: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP4 (uint8 packed nibbles + uint8 UE8M0 scales) to BF16.

    `upcast_from_mxfp` is a Triton kernel and needs CUDA tensors. The
    dumped checkpoint tensors are loaded to CPU; move to CUDA, then
    bring the result back to CPU for the rest of the analysis (which
    does plain torch matmuls in float32).
    """
    from triton_kernels.numerics_details.mxfp import upcast_from_mxfp
    w = w_uint8.cuda() if not w_uint8.is_cuda else w_uint8
    s = s_uint8.cuda() if not s_uint8.is_cuda else s_uint8
    out = upcast_from_mxfp(w, s, target_dtype=torch.bfloat16, axis=-1)
    return out.cpu()


def fp8_dequant(fp8_active: torch.Tensor, scale_active: torch.Tensor,
                 group_size: int) -> torch.Tensor:
    """Dequantize FP8 e4m3 with per-group FP32 or packed-UE8M0-INT32 scales.

    `fp8_active` shape: [..., K]. `scale_active` shape:
    [..., K // group_size] when FP32, or [..., K // (group_size*4)]
    when packed INT32 (4 UE8M0 bytes per int32).
    """
    fp8_f = fp8_active.float()
    if scale_active.dtype == torch.float32:
        s = scale_active
    elif scale_active.dtype in (torch.int32, torch.int):
        s = _unpack_ue8m0_int32_to_fp32(scale_active)
    else:
        raise NotImplementedError(
            f"unsupported FP8 scale dtype {scale_active.dtype}"
        )
    prefix = s.shape[:-1]
    ng = s.shape[-1]
    s = s.unsqueeze(-1).expand(*prefix, ng, group_size)
    s = s.reshape(*prefix, ng * group_size)
    return fp8_f * s


def stage_post_gemm1(stages: dict, raw_weights_per_expert: dict,
                      inputs: dict) -> dict:
    """Compare gateup_output_active vs BF16(hidden @ w13.T) per active expert,
    using the live captured FP8 hidden activations dequantized to BF16.

    We use the captured hidden_states_active+scale (post-quant) as the
    reference input rather than the pre-quant BF16 hidden_states because
    deepgemm sees the FP8-quantized side.
    """
    g1 = stages["post_gemm1"]
    active = g1["active_experts"].long()
    masked_m = g1["masked_m"]
    fp8 = g1["hidden_states_active"]
    sf = g1["hidden_states_scale_active"]
    gateup = g1["gateup_output_active"]

    # Dequantize FP8 hidden to BF16. Group size = 128 per the live path.
    hid_bf16 = fp8_dequant(fp8, sf, group_size=128).to(torch.bfloat16)
    diffs = []
    for i, e in enumerate(active.tolist()):
        m = int(masked_m[e].item())
        if m == 0:
            continue
        w = raw_weights_per_expert[e]["w13_w"]
        s = raw_weights_per_expert[e]["w13_s"]
        wb = bf16_dequant_w(w, s).float()           # [N, K]
        ref = hid_bf16[i, :m].float() @ wb.T        # [m, N]
        test = gateup[i, :m].float()
        diffs.append((e, alpha_metrics(test, ref)))
    return {"per_expert": diffs}


def stage_post_gemm2(stages: dict, raw_weights_per_expert: dict) -> dict:
    """Compare down_output_active vs BF16(silu(gate)*up @ w2.T) two ways:
    against ideal silu(gate)*up, and against the actually-quantized
    activation that GEMM2 received. Splitting these isolates the
    activation bridge from GEMM2 itself.
    """
    g1 = stages["post_gemm1"]
    pa = stages["post_activation"]
    pg2 = stages.get("pre_gemm2")  # post-scale-prep dump
    g2 = stages["post_gemm2"]
    active = g1["active_experts"].long()
    masked_m = g1["masked_m"]
    gateup = g1["gateup_output_active"]
    down_in = pa["down_input_active"]
    sf_pre = pa["down_input_scale_active"]
    sf_post = pg2["down_input_scale_active"] if pg2 is not None else sf_pre
    down_out = g2["down_output_active"]
    diffs = []
    for i, e in enumerate(active.tolist()):
        m = int(masked_m[e].item())
        if m == 0:
            continue
        gu = gateup[i, :m].float()
        N = gu.shape[-1]
        gate, up = gu[:, : N // 2], gu[:, N // 2 :]
        silu_ref = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        di_pre = fp8_dequant(down_in[i:i+1, :m], sf_pre[i:i+1, :m],
                              group_size=128)[0]
        di_post = fp8_dequant(down_in[i:i+1, :m], sf_post[i:i+1, :m],
                               group_size=128)[0]
        act_pre = alpha_metrics(di_pre, silu_ref)
        act_post = alpha_metrics(di_post, silu_ref)
        pre_vs_post = alpha_metrics(di_post, di_pre)
        w = raw_weights_per_expert[e]["w2_w"]
        s = raw_weights_per_expert[e]["w2_s"]
        wb = bf16_dequant_w(w, s).float()
        ref_gemm2_ideal = silu_ref.float() @ wb.T
        # The "real" GEMM2 reference: use the POST-prep dequantized
        # activation (the layout GEMM2 actually consumed).
        ref_gemm2_post = di_post.float() @ wb.T
        test_gemm2 = down_out[i, :m].float()
        gemm2_vs_ideal = alpha_metrics(test_gemm2, ref_gemm2_ideal)
        gemm2_vs_post = alpha_metrics(test_gemm2, ref_gemm2_post)
        diffs.append({
            "expert": e,
            "act_pre_scale": act_pre,
            "act_post_scale": act_post,
            "pre_vs_post_scale": pre_vs_post,
            "gemm2_vs_ideal": gemm2_vs_ideal,
            "gemm2_vs_post_deq_input": gemm2_vs_post,
        })
    return {"per_expert": diffs}


def _assert_src2dst(dst: int, e_global: int, M_pad: int, src: int) -> int:
    """Validate src2dst contract: dst // M_pad == expert; return row."""
    e_from_dst = dst // M_pad
    if e_from_dst != e_global:
        raise AssertionError(
            f"src2dst expert mismatch: src={src} topk_e={e_global} "
            f"dst={dst} dst_e={e_from_dst} M_pad={M_pad}"
        )
    return dst % M_pad


def stage_post_reorder(stages: dict) -> dict:
    """Verify post_reorder_triton_kernel matches manual combine."""
    if "pre_post_reorder" not in stages or "post_reorder_before_rsf" not in stages:
        return {"skip": "missing stages"}
    pp = stages["pre_post_reorder"]
    src2dst = pp["src2dst"].long()
    topk_ids = pp["topk_ids"].long()
    topk_weights = pp["topk_weights"].float()
    top_k = int(pp["top_k"])
    H = pp["hidden_states_shape"][1]
    B = pp["hidden_states_shape"][0]
    runner_out = pp["runner_output_active"].float()  # [E_active, M_pad, H]
    active = stages["post_gemm1"]["active_experts"].long()
    masked_m = stages["post_gemm1"]["masked_m"].long()
    M_pad = stages["post_gemm1"]["m_pad"]
    # Build dst → (e_active_idx, row) mapping.
    expert_idx_by_global = {int(e): i for i, e in enumerate(active.tolist())}
    out_ref = torch.zeros(B, H, dtype=torch.float32)
    flat_topk_ids = topk_ids.flatten()
    for src in range(flat_topk_ids.numel()):
        e_global = int(flat_topk_ids[src])
        if e_global < 0:
            continue
        if e_global not in expert_idx_by_global:
            # Expert not on this rank — no contribution from this slot.
            continue
        dst = int(src2dst.flatten()[src])
        if dst < 0:
            continue
        e_local_idx = expert_idx_by_global[e_global]
        row_in_expert = _assert_src2dst(dst, e_global, M_pad, src)
        if row_in_expert >= int(masked_m[e_global]):
            continue
        token = src // top_k
        slot = src % top_k
        assert token < B and slot < top_k
        w = float(topk_weights[token, slot])
        out_ref[token] += w * runner_out[e_local_idx, row_in_expert]
    test_before = stages["post_reorder_before_rsf"]["output"].float()
    return {"manual_vs_kernel": alpha_metrics(test_before, out_ref)}


def stage_rsf(stages: dict) -> dict:
    if "post_reorder_after_rsf" not in stages:
        return {"skip": "missing stage"}
    before = stages["post_reorder_before_rsf"]["output"]
    after = stages["post_reorder_after_rsf"]["output"]
    rsf = stages["pre_post_reorder"]["routed_scaling_factor"] or 1.0
    return alpha_metrics(after, before * rsf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--gate-up", default="gate_first_half",
                    choices=GATE_UP_CONVENTIONS,
                    help="w13 gate/up convention assumed for the BF16 reference")
    p.add_argument("--sweep-gate-up", action="store_true",
                    help="sweep all 6 gate/up conventions at post_gemm1 and "
                         "select lowest rel_l2 winner for downstream stages")
    args = p.parse_args()

    print(f"loading {args.stages}")
    stages = torch.load(args.stages, map_location="cpu", weights_only=False)
    print(f"loading {args.inputs}")
    inputs = torch.load(args.inputs, map_location="cpu", weights_only=False)
    dumpout = None
    if args.output:
        print(f"loading {args.output}")
        dumpout = torch.load(args.output, map_location="cpu", weights_only=False)
    # Sanity: stage / input routing tensors must agree (same forward).
    # These are HARD asserts to prevent silent cross-PID file mixing.
    if "topk_ids" in inputs and "pre_post_reorder" in stages:
        a = inputs["topk_ids"]
        b = stages["pre_post_reorder"]["topk_ids"]
        assert torch.equal(a, b), (
            "inputs/topk_ids != stages/topk_ids — dumps from different forwards"
        )
    if "topk_weights" in inputs and "pre_post_reorder" in stages:
        a = inputs["topk_weights"].float()
        b = stages["pre_post_reorder"]["topk_weights"].float()
        assert torch.allclose(a, b, rtol=0, atol=0), (
            "inputs/topk_weights != stages/topk_weights — dumps from different forwards"
        )
    if dumpout is not None and "post_reorder_after_rsf" in stages:
        assert "deepgemm_output" in dumpout, "dumpout missing deepgemm_output"
        final_stage = stages["post_reorder_after_rsf"]["output"].float()
        final_apply = dumpout["deepgemm_output"].float()
        assert final_stage.shape == final_apply.shape, (
            f"shape mismatch: stages.final {final_stage.shape} vs "
            f"apply().output {final_apply.shape}"
        )
        sanity = alpha_metrics(final_stage, final_apply)
        # BF16 storage round-trip: allow tiny drift; reject anything
        # significant which indicates dumps from different forwards.
        assert sanity["rel_l2"] < 1e-4, (
            f"stages.final != apply().output: {sanity} — different forwards"
        )
        print(f"\nsanity: stages.final vs apply().output: {sanity}")

    print("\nstages present:")
    for k in stages:
        print(f"  {k}: {sorted(stages[k].keys())}")
    print("\ninputs:", sorted(inputs.keys()))

    # Active expert IDs from the staged dump.
    active_ids = stages["post_gemm1"]["active_experts"].long().tolist()
    print(f"\nactive experts: {active_ids}")

    # Load raw MXFP4 checkpoint weights for these experts.
    print(f"loading checkpoint weights for {len(active_ids)} experts "
          f"(rank={args.rank}, tp={args.tp})")
    rank_per_expert = {}
    for e in active_ids:
        full = _load_expert_weights(e)
        rank_w = _rank_shard_components(full, rank=args.rank, tp=args.tp)
        rank_per_expert[e] = rank_w

    def _build(per_expert, convention):
        out = {}
        for e, rw in per_expert.items():
            w13_w, w13_s = build_w13(rw, convention)
            out[e] = {"w13_w": w13_w.cpu(), "w13_s": w13_s.cpu(),
                      "w2_w":  rw["w2_w"].cpu(), "w2_s":  rw["w2_s"].cpu()}
        return out

    if args.sweep_gate_up:
        print("\n=== gate/up convention sweep at post_gemm1 ===")
        best = None
        for cv in GATE_UP_CONVENTIONS:
            raw = _build(rank_per_expert, cv)
            try:
                g1 = stage_post_gemm1(stages, raw, inputs)
            except Exception as ex:
                print(f"  {cv:>20}: ERR {type(ex).__name__}: {ex}")
                continue
            mean_l2 = sum(m["rel_l2"] for _, m in g1["per_expert"]) / max(
                1, len(g1["per_expert"]))
            print(f"  {cv:>20}: mean rel_l2={mean_l2:.3e}")
            if best is None or mean_l2 < best[1]:
                best = (cv, mean_l2)
        print(f"selected gate_up={best[0]} (mean L2={best[1]:.3e})")
        gate_up = best[0]
    else:
        gate_up = args.gate_up

    raw_per_expert = _build(rank_per_expert, gate_up)

    print(f"\n=== STAGE 1: post_gemm1 (gateup_output, gate_up={gate_up}) ===")
    g1 = stage_post_gemm1(stages, raw_per_expert, inputs)
    for e, m in g1["per_expert"]:
        print(f"  expert {e}: α={m['alpha']:.4f}  L2={m['rel_l2']:.3e}  "
              f"resid={m['rel_resid_after_alpha']:.3e}  n={m['n']}")

    print("\n=== STAGE 2/3: post_activation + pre_gemm2 + post_gemm2 ===")
    g2 = stage_post_gemm2(stages, raw_per_expert)
    for r in g2["per_expert"]:
        ap = r["act_pre_scale"]
        aP = r["act_post_scale"]
        pp = r["pre_vs_post_scale"]
        gi = r["gemm2_vs_ideal"]
        gp = r["gemm2_vs_post_deq_input"]
        print(f"  expert {r['expert']}:")
        print(f"    act_pre  α={ap['alpha']:.4f} L2={ap['rel_l2']:.3e}")
        print(f"    act_post α={aP['alpha']:.4f} L2={aP['rel_l2']:.3e}")
        print(f"    pre_vs_post_scale α={pp['alpha']:.4f} L2={pp['rel_l2']:.3e}")
        print(f"    gemm2|ideal α={gi['alpha']:.4f} L2={gi['rel_l2']:.3e}")
        print(f"    gemm2|post  α={gp['alpha']:.4f} L2={gp['rel_l2']:.3e}")

    print("\n=== STAGE 4: post_reorder (manual combine vs kernel) ===")
    pr = stage_post_reorder(stages)
    print(f"  {pr}")

    print("\n=== STAGE 5: post_reorder_after_rsf vs before_rsf*rsf ===")
    rsf = stage_rsf(stages)
    print(f"  {rsf}")

    print("\n=== Verdict ===")
    print("Locate the FIRST stage where α first deviates substantially "
          "from 1.0; that stage is the source of the live magnitude error.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
