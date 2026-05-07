# Phase 11: sm_120 MoE matmul_ogs constraint tuning

This directory archives the empirical evidence behind the sm_120 MoE
matmul_ogs constraints set in
`python/sglang/srt/layers/quantization/mxfp4.py:_swizzle_mxfp4`
for V4-Flash on Blackwell consumer / workstation cards (sm_120, e.g.
RTX PRO 6000 Blackwell).

## Background

V4-Flash routed experts go through the OpenAI `triton_kernels.matmul_ogs`
library (vendored at `/usr/lib/python3.14/site-packages/triton_kernels/`).
The library uses a deterministic heuristic at
`triton_kernels/matmul_ogs_details/opt_flags_details/opt_flags_nvidia.py`
to pick `(block_m, block_n, block_k, num_warps, num_stages,
is_persistent, ...)` for each matmul shape.

The heuristic was authored with sm_90 / sm_100 in mind. It treats every
`compute_capability ≥ 10` device the same — sm_100 (B200 datacenter)
and sm_120 (RTX PRO 6000 desktop/workstation) are not distinguished
even though they have very different shared-memory budgets (228 KB vs
99 KB) and different Tensor-Core sweet spots.

Public V4-Flash deployments on H100/B200 don't need a separate tuning
pass because the heuristic happens to produce reasonable picks for
those architectures (and where it doesn't, sglang ships JSON
overrides on a different code path that V4-Flash MXFP4 doesn't use).
On sm_120 nobody had verified what the heuristic picks. This sweep
closed that gap.

## Captured live shapes

V4-Flash on sm_120 with `--cuda-graph-bs [1,2,4,8]` exercises five
distinct M values per layer per forward, all sharing the same weights:

| M | K | w1.shape | w2.shape | dtype | n_active |
|---|---|---|---|---|---|
| 1, 2, 4, 6, 8 | 4096 | (256, 4096, 2048) | (256, 1024, 4096) | MXFP4 (FP4 + e8m0 scales) | 8 |

Captured via a one-shot per-shape trace at `triton_kernel_fused_experts`
on a real decode forward (see Phase 11.0 in the project plan).

## Sweep design

- `is_persistent` locked False on sm_120 (no TMA MXFP4 path).
- `block_n`, `num_warps`, `group_m`, `xcd_swizzle` are heuristic-derived
  by the library and **cannot** be overridden via the constraints API.
  Verified by reading
  `triton_kernels/matmul_ogs_details/opt_flags.py:166-291`.
- `block_k=256` rejected as a sweep candidate because it doubles smem
  vs the existing baseline 128 and is high-risk against the 99 KB
  sm_120 budget. `block_k=64` halves smem (safe-low) and is included.
- `epilogue_subtile` left to the heuristic.

Effective sweep grid: `block_m ∈ {16, 32}` × `block_k ∈ {64, 128}` ×
`num_stages ∈ {1, 2, 3, 4}` = 16 candidates plus the `NO_OVERRIDE`
baseline = 17 total.

Methodology (per the project plan's wall-clock gate):

- Each candidate restarts the service via `systemctl restart`.
- Health-poll until ready (or fast-fail on `is-failed`).
- One discarded warmup decode (256 tokens, `ignore_eos=True`).
- Three measurement decodes (256 tokens each, `temperature=0`,
  `ignore_eos=True`); record each tok/s.
- Wall-clock metric: mean tok/s over the three measurements.

## Results

See `sweep_results.csv`. Headline:

| Candidate | Mean tok/s | Δ vs baseline |
|---|---|---|
| `NO_OVERRIDE` | 15.408 | — |
| `block_m=32, block_k=64, num_stages=1` (winner) | **18.894** | **+22.6%** |
| `block_m=32, block_k=64, num_stages=2` (tied) | 18.894 | +22.6% |
| `block_m=16, block_k=128, num_stages=2` (worst) | 9.344 | −39.4% |

Per-candidate run-spread was always < 0.04 tok/s — far below the
~0.3 tok/s noise floor we expected on this hardware. Confidence in
the winner is high.

Two qualitative observations from the full data:

1. `block_m=32` is universally better than `block_m=16` for this MoE
   shape on sm_120. All eight `block_m=32` candidates landed in the
   18.5-18.9 tok/s range; all eight `block_m=16` candidates were
   ≤17.9 tok/s.
2. With `block_m=16` and `num_stages > 1`, performance collapses
   into the 9-14 tok/s range — almost certainly register spills or
   another bad heuristic interaction. The pre-Phase-11 baseline
   (`block_m=16, block_k=128, num_stages=1`) sits one knob away
   from that disaster zone.

## Landed configuration

```python
constraints = {
    "is_persistent": False,
    "block_m": 32,
    "block_k": 64,
    "num_stages": 1,
}
```

Set in `python/sglang/srt/layers/quantization/mxfp4.py:_swizzle_mxfp4`
under the `is_sm120_supported()` branch.

## Reproducing the sweep

`sweep_driver.py` is the driver used. It requires:

- The `SGLANG_SM120_MOE_CONSTRAINTS` env-var hook in `mxfp4.py` (this
  hook was removed after Phase 11 closed; it would need to be
  re-added to `_swizzle_mxfp4` if a future re-tune is wanted).
- A live `sglang@deepseek_v4_flash` systemd service.
- Root (writes `/etc/sglang/deepseek_v4_flash.env`, calls
  `systemctl restart`).

Run as `sudo python sweep_driver.py`. Logs to `/home/will/phase11_sweep.csv`.
~50 minutes wall-clock for the 17-candidate grid.
