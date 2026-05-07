#!/usr/bin/env python3
"""Phase 11.3 — sm_120 MoE matmul_ogs constraint sweep driver.

For each candidate constraint override:

  1. Writes /etc/sglang/deepseek_v4_flash.env (compact JSON value).
  2. systemctl restart sglang@deepseek_v4_flash.
  3. Polls /health until ready (timeout 5 min, fast-fail on `is-failed`).
  4. Sends one warmup decode (256 tokens, ignore_eos=True; discarded).
  5. Sends 3 measurement decodes; records each tok/s.
  6. Logs candidate + results to /home/will/phase11_sweep.csv.
  7. Continues to next candidate even if a candidate fails.

Run as root (needs to write /etc/sglang/ and call systemctl). The script
restores the env file to its pre-sweep contents on exit. On startup, if a
stale backup is found from a prior interrupted run, it is restored and the
script exits — the user must rerun explicitly.

Output CSV columns:
  idx, candidate, run1_tps, run2_tps, run3_tps,
  mean_tps, min_tps, max_tps, elapsed_s, status
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ENV_FILE = Path("/etc/sglang/deepseek_v4_flash.env")
ENV_BACKUP = Path("/etc/sglang/deepseek_v4_flash.env.phase11_backup")
LOG_FILE = Path("/home/will/phase11_sweep.csv")
SERVICE = "sglang@deepseek_v4_flash"
HEALTH_URL = "http://127.0.0.1:30000/health"
GENERATE_URL = "http://127.0.0.1:30000/generate"
PROMPT = "Write a short essay about why mathematicians like prime numbers."
MAX_NEW_TOKENS = 256
READY_TIMEOUT_S = 300
WARMUP_TIMEOUT_S = 180
MEASURE_TIMEOUT_S = 120

# Candidate space, given the Phase 11.1 audit:
#   - is_persistent locked False on sm_120 (no TMA MXFP4 path).
#   - block_n / num_warps / group_m / xcd_swizzle are heuristic-derived
#     by the OAI library; constraints API CANNOT override them.
#   - So the override-able knobs we sweep are block_m, block_k, num_stages,
#     epilogue_subtile.
#   - block_k=128 was set originally for the smem budget; block_k=64 halves
#     the smem footprint (safe-low) and we keep it. block_k=256 doubles smem
#     usage and is high-risk on the 99 KB sm_120 budget; dropped.
#   - epilogue_subtile is left to the heuristic — sweeping it across all
#     candidates is low-impact for this MoE shape and not worth the
#     candidate-count multiplier.
CANDIDATES: list[dict | None] = [None]  # row 1 = TRUE NO-OVERRIDE BASELINE
CANDIDATES.extend(
    {"block_m": bm, "block_k": bk, "num_stages": ns}
    for bm in (16, 32)
    for bk in (64, 128)
    for ns in (1, 2, 3, 4)
)


def must_be_root():
    if os.geteuid() != 0:
        sys.stderr.write("ERROR: must run as root (sudo).\n")
        sys.exit(1)


def fmt_float(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else ""


_OVERRIDE_PREFIX = "SGLANG_SM120_MOE_CONSTRAINTS="


def write_env_for(constraint: dict | None):
    """Preserve original env-file contents; only edit the override line."""
    source = ENV_BACKUP if ENV_BACKUP.exists() else ENV_FILE
    lines = source.read_text().splitlines() if source.exists() else []
    filtered = [
        line for line in lines
        if not line.lstrip().startswith(_OVERRIDE_PREFIX)
    ]
    if constraint is not None:
        # Compact JSON: no spaces — systemd EnvironmentFile treats VAR=value
        # as a single token and chokes on embedded whitespace.
        payload = json.dumps(constraint, separators=(",", ":"))
        filtered.append(f"{_OVERRIDE_PREFIX}{payload}")
    ENV_FILE.write_text("\n".join(filtered) + "\n")


def backup_env_file():
    if ENV_FILE.exists() and not ENV_BACKUP.exists():
        shutil.copy2(ENV_FILE, ENV_BACKUP)


def restore_env_file():
    if ENV_BACKUP.exists():
        shutil.copy2(ENV_BACKUP, ENV_FILE)
        ENV_BACKUP.unlink()
    else:
        # No backup means there was no pre-sweep env file. Best effort:
        # clear the override line by writing the no-override variant.
        write_env_for(None)


def restart_service():
    subprocess.run(
        ["systemctl", "reset-failed", SERVICE],
        check=False,
        capture_output=True,
    )
    subprocess.run(
        ["systemctl", "restart", SERVICE],
        check=True,
        capture_output=True,
    )


def service_failed() -> bool:
    return (
        subprocess.run(
            ["systemctl", "is-failed", "--quiet", SERVICE],
            check=False,
        ).returncode
        == 0
    )


def poll_ready(timeout_s: int) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if service_failed():
            return False
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    return False


def measure_decode_tps(timeout_s: int = MEASURE_TIMEOUT_S) -> float:
    """One greedy 256-token decode (forced via ignore_eos). Returns tok/s."""
    body = json.dumps(
        {
            "text": PROMPT,
            "sampling_params": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        GENERATE_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        result = json.loads(r.read())
    meta = result["meta_info"]
    tokens = meta["completion_tokens"]
    if tokens != MAX_NEW_TOKENS:
        raise RuntimeError(
            f"expected {MAX_NEW_TOKENS} completion tokens, got {tokens}"
        )
    return tokens / meta["e2e_latency"]


def run_candidate(constraint: dict | None) -> dict:
    """Run one candidate. Returns result dict with runs/mean/min/max/status."""
    write_env_for(constraint)
    try:
        restart_service()
    except subprocess.CalledProcessError as e:
        return {"status": f"RESTART_FAIL:{e.returncode}"}
    if not poll_ready(READY_TIMEOUT_S):
        if service_failed():
            return {"status": "SERVICE_FAILED"}
        return {"status": "NOT_READY"}
    try:
        measure_decode_tps(timeout_s=WARMUP_TIMEOUT_S)
    except Exception as e:
        return {"status": f"WARMUP_FAIL:{type(e).__name__}"}
    runs: list[float | None] = []
    for i in range(3):
        try:
            runs.append(measure_decode_tps())
        except Exception as e:
            runs.append(None)
            print(f"  measure {i+1} failed: {e}", flush=True)
    valid = [r for r in runs if isinstance(r, (int, float))]
    if not valid:
        return {"runs": runs, "status": "ALL_RUNS_FAILED"}
    # Plan's wall-clock gate requires ≥3 measurements after warmup. A
    # candidate with partial runs gets a non-OK status so winner-picking
    # ignores it.
    status = "OK" if len(valid) == 3 else f"MEASURE_PARTIAL:{len(valid)}/3"
    return {
        "runs": runs,
        "mean": sum(valid) / len(valid),
        "min": min(valid),
        "max": max(valid),
        "status": status,
    }


def append_csv_row(idx: int, candidate, result: dict, elapsed_s: float):
    runs = result.get("runs", [None, None, None])
    while len(runs) < 3:
        runs.append(None)
    label = "NO_OVERRIDE" if candidate is None else json.dumps(
        candidate, separators=(",", ":")
    )
    row = [
        idx,
        label,
        fmt_float(runs[0]),
        fmt_float(runs[1]),
        fmt_float(runs[2]),
        fmt_float(result.get("mean")),
        fmt_float(result.get("min")),
        fmt_float(result.get("max")),
        f"{elapsed_s:.1f}",
        result["status"],
    ]
    with LOG_FILE.open("a", newline="") as f:
        csv.writer(f).writerow(row)


def main():
    must_be_root()

    if ENV_BACKUP.exists():
        sys.stderr.write(
            "[sweep] stale backup found from previous interrupted run; "
            "restoring and exiting. Verify state, then rerun.\n"
        )
        restore_env_file()
        try:
            restart_service()
        except Exception as e:
            sys.stderr.write(f"[sweep] WARN: restart after restore failed: {e}\n")
        sys.exit(1)

    backup_env_file()

    interrupted = {"flag": False}

    def handle_signal(*_):
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "idx",
                    "candidate",
                    "run1_tps",
                    "run2_tps",
                    "run3_tps",
                    "mean_tps",
                    "min_tps",
                    "max_tps",
                    "elapsed_s",
                    "status",
                ]
            )

    print(f"[sweep] {len(CANDIDATES)} candidates")
    print(f"[sweep] log: {LOG_FILE}")
    started = time.time()
    try:
        for idx, cfg in enumerate(CANDIDATES, start=1):
            if interrupted["flag"]:
                print("\n[sweep] interrupt received; stopping cleanly")
                break
            label = "NO_OVERRIDE" if cfg is None else json.dumps(
                cfg, separators=(",", ":")
            )
            print(f"\n[sweep] {idx}/{len(CANDIDATES)} — {label}", flush=True)
            t0 = time.time()
            result = run_candidate(cfg)
            elapsed = time.time() - t0
            mean_str = fmt_float(result.get("mean")) or "—"
            print(
                f"[sweep] {idx}/{len(CANDIDATES)} — "
                f"status={result['status']} mean={mean_str} ({elapsed:.1f}s)",
                flush=True,
            )
            append_csv_row(idx, cfg, result, elapsed)
    finally:
        elapsed_total = time.time() - started
        print(
            f"\n[sweep] total: {elapsed_total/60:.1f} min — "
            f"restoring pre-sweep env",
            flush=True,
        )
        restore_env_file()
        try:
            restart_service()
        except Exception as e:
            print(f"[sweep] WARN: final restart failed: {e}")


if __name__ == "__main__":
    main()
