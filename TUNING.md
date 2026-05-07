# sm_120 V4-Flash tuning record

This file is the canonical log of empirical performance tuning done for
DeepSeek-V4-Flash on sm_120 (RTX PRO 6000 Blackwell). Each entry
captures the methodology, the git commit hash that landed the result,
and a pointer to the raw evidence so any of it can be reproduced.

## Hardware / software baseline

- 2× NVIDIA RTX PRO 6000 Blackwell (sm_120, 96 GB VRAM each, no NVLink, PCIe).
- TP=2.
- sglang fork at branch `wjh/v4-flash-mxfp4-routed-experts`.
- Triton 3.5.1, sgl-kernel 0.4.2.post1, transformers v5.6.0,
  python-tilelang 0.1.9.

## Currently-active tuning artifacts

| Tuning lever | Landed at | Live tok/s impact | Methodology / data |
|---|---|---|---|
| Phase 7.5 — `_w8a8_block_fp8_matmul` JSON configs | `0adf6fb` | (used by FP8 paths in shared experts / attention; per-shape baseline win, not separately measured at decode level) | 7 JSON tile configs in `python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1/` |
| Phase 11 — `matmul_ogs` MoE constraints (V4-Flash MXFP4 routed experts) | `0299f13` | **15.41 → 18.89 tok/s (+22.6%)** | `python/sglang/srt/layers/quantization/sm120_mxfp4_tuning/` |
| Phase 14 (out of scope here, listed for context) — full upstream merge | `43d710b`, `1bb48eb`, `8ee8a8f`, `82c45e8`, `60090c2` | Parity build, no perf claim | — |

Other phases (Phase 4–10) closed correctness gaps, not perf tuning.
See `nested-growing-hammock.md` and the project README for that context.

## Phase 11 — sm_120 MoE matmul_ogs constraint tuning

**Scope caveat:** `update_opt_flags_constraints` is a process-global
dict in the OAI `triton_kernels` library. The constraints landed for
V4-Flash apply to any sm_120 MXFP4 swizzle in the process, not only
V4-Flash. Values validated only on V4-Flash routed-expert shapes
(E=256, K=4096, N∈{2048, 1024}, M∈{1..8}). Re-run the sweep before
using this fork to serve a different sm_120 MXFP4 MoE model.

**Landing commit:** `0299f13`  
**Sweep harness commit (later removed):** `0588e90`  
**Trace-probe commits:** `0a991cd` (added) → `ab64457` (reverted)  
**Raw data:** `python/sglang/srt/layers/quantization/sm120_mxfp4_tuning/sweep_results.csv`  
**Sweep driver:** `python/sglang/srt/layers/quantization/sm120_mxfp4_tuning/sweep_driver.py`  
**README:** `python/sglang/srt/layers/quantization/sm120_mxfp4_tuning/README.md`

### Why

V4-Flash routed experts go through OpenAI's `triton_kernels.matmul_ogs`
library. Its config heuristic at
`triton_kernels/matmul_ogs_details/opt_flags_details/opt_flags_nvidia.py`
treats every `compute_capability ≥ 10` device the same — sm_120
desktop Blackwell falls under the sm_100 datacenter branch despite
having less than half the shared-memory budget. No sm_120-specific
tuning existed in the merged tree. Public V4-Flash deployments on
H100/B200 don't need this work because the heuristic happens to
produce reasonable picks for them.

### Methodology

1. **Phase 11.0** — capture the live MoE shapes. One-shot trace at
   `triton_kernel_fused_experts(...)`. Found:
   `M ∈ {1, 2, 4, 6, 8}, K=4096, w1=(256, 4096, 2048), w2=(256, 1024, 4096)`,
   MXFP4 weights, 8 active experts. Trace probe added in `0a991cd`
   and reverted in `ab64457`.
2. **Phase 11.1** — audited the constraints API
   (`triton_kernels/matmul_ogs_details/opt_flags.py:166-291`).
   Verified that `block_m`, `block_k`, `num_stages`, `is_persistent`,
   `split_k`, `epilogue_subtile`, `idle_sms`, `max_allowable_mn` are
   override-able; `block_n`, `num_warps`, `group_m`, `xcd_swizzle`
   are heuristic-derived and cannot be overridden.
3. **Phase 11.2/11.3** — restart-per-candidate sweep across
   `block_m ∈ {16, 32}` × `block_k ∈ {64, 128}` ×
   `num_stages ∈ {1, 2, 3, 4}` (16 explicit candidates) plus a
   `NO_OVERRIDE` baseline = 17 candidates. Each candidate restarted
   the service via `systemctl`, ran one warmup decode + 3 measurement
   decodes (256 greedy tokens, `ignore_eos=True`), captured mean
   tok/s. ~50 minutes total wall-clock.
4. **Phase 11.5** — landed the winner
   (`block_m=32, block_k=64, num_stages=1`) into the static sm_120
   constraint dict in
   `python/sglang/srt/layers/quantization/mxfp4.py:_swizzle_mxfp4`.
   Removed the `SGLANG_SM120_MOE_CONSTRAINTS` env-var sweep hook.

### Reproducing

The sweep driver lives at
`python/sglang/srt/layers/quantization/sm120_mxfp4_tuning/sweep_driver.py`.
To re-run (e.g. after a Triton or triton_kernels upgrade, or after a
checkpoint format change):

1. Add the sweep override hook back to
   `python/sglang/srt/layers/quantization/mxfp4.py:_swizzle_mxfp4`
   (the form is documented in the diff at `0588e90`):
   ```python
   import json as _json
   import os as _os
   _override = _os.environ.get("SGLANG_SM120_MOE_CONSTRAINTS")
   if _override:
       constraints.update(_json.loads(_override))
   ```
2. Build + install + run as `sudo python sweep_driver.py`.
3. Inspect `/home/will/phase11_sweep.csv` (or whatever `LOG_FILE` points at).
4. Update the static constraint dict with the new winner. Remove the
   override hook.
5. Update this TUNING.md with the new commit hash and tok/s delta.

The sweep driver writes/restores `/etc/sglang/deepseek_v4_flash.env`
to avoid clobbering site-specific settings.

## Other tuning axes still on the table

These have **not** been closed. Listed for future work:

- **Phase 12 — sparse-MLA Tensor-Core-friendly rewrite.** Structural
  not pure-constant. Bigger lever than any remaining sweep. See
  `nested-growing-hammock.md` for the plan.
- **Sweeping our own Triton kernels' constants.** `_fp8_paged_mqa_logits_kernel`
  (~0.7% of decode), MHC kernels (~few%), `hadamard_transform`
  variants. Each is a small slice of decode time; combined possibly
  5-10%, worth doing only if Phase 12 underperforms.
- **More tuned JSON configs for `_w8a8_block_fp8_matmul`** if the
  journal still warns about default configs at startup. Phase 7.5
  covered the dominant shapes; gaps may remain.

## Anti-patterns burned in earlier phases

(Documented here so they don't get re-attempted blindly.)

- **Phase 8.1 — `BLOCK_DV` 128 → 256 in `_sparse_mla_decode_kernel`.**
  Harness passed, no PTX register spills, but live decode regressed
  33%. Reverted in commit `d903476`. Lesson: harness pass + zero spills
  is necessary but not sufficient — the wall-clock measurement is the
  gate.
- **EAGLE-V2 with V4-Flash.** Hits an architectural mismatch in
  upstream sglang (V4 uses unified compressed KV; EAGLE expects split
  K/V). Not sm_120-specific. Documented in plan; do not re-attempt
  without a matching upstream sglang change.
- **NVFP4 routed experts for V4-Flash.** Checkpoint-blocked: V4-Flash
  ships only as MXFP4. The original parity roadmap incorrectly
  claimed H100/B200 deployments use NVFP4 routed experts.
