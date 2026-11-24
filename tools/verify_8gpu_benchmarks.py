#!/usr/bin/env python3
"""Offline verifier for the eight-GPU DeepSeek-V4.1-Flash-1017 repository.

Checks numeric invariants, Speed9 decode/prefill gains, 10x1M/24x1M capacity,
quality gates, sanitization hygiene, and strict 8-GPU-only content isolation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "data/2026-11-22-eight-gpu-speed9.json").read_text(encoding="utf-8"))
CFG = json.loads((ROOT / "configs/2026-11-22-eight-gpu-speed9.json").read_text(encoding="utf-8"))

# 1. Core 8-GPU configuration invariants
assert CFG["tp"] == 4 and CFG["pp"] == 2 and CFG["layer_partition"] == [20, 20]
assert CFG["kv_GiB_per_gpu"] == 36 and CFG["max_num_seqs"] == 24 and CFG["max_model_len"] == 1048576
assert CFG["environment"]["VLLM_LANGCHAO_BMM_MAX_T"] == "48"
assert CFG["environment"]["VLLM_MOET_PREFILL_PIPE"] == "2"
assert CFG["environment"]["VLLM_MOET_PREFILL_SP_RS"] == "nccl"
assert CFG["environment"]["VLLM_MOET_PREFILL_SP_GATHER"] == "mxfp8"
assert CFG["environment"]["VLLM_MOET_PREFILL_PIPE_ORDER"] == "comm2prio"
assert CFG["environment"]["VLLM_MOET_MOE_MEGA"] == "0"
assert CFG["image_content_id"] == DATA["runtime_identities"]["selected_cand4_20261122"]["image_content_id"]
assert CFG["inference_fingerprint"] == DATA["runtime_identities"]["selected_cand4_20261122"]["inference_fingerprint"]

# 2. C1 step times and decode gains vs baseline
t0 = DATA["short_prompt_t0"]
for case in ("chinese", "code", "english"):
    b = t0["baseline"]["c1"][case]
    b_ms = 1000.0 * b["eff"] / b["tps"]
    assert 10.0 <= b_ms <= 10.3, (case, b_ms)
    for rep in ("cand4_repeat1", "cand4_repeat2"):
        r = t0[rep]["c1"][case]
        r_ms = 1000.0 * r["eff"] / r["tps"]
        assert r["tps"] > b["tps"] * 1.08, (rep, case, r["tps"], b["tps"])
        assert r_ms < b_ms * 0.90, (rep, case, r_ms, b_ms)

# 3. Cold prefill reductions > 40%
pf = DATA["cold_prefill_single"]
assert pf["tokens_163k"]["passed"] and pf["tokens_1m"]["passed"]
r163 = 1.0 - pf["tokens_163k"]["cand4_repeat2_median_s"] / pf["tokens_163k"]["baseline_median_s"]
r1m = 1.0 - pf["tokens_1m"]["cand4_repeat2_median_s"] / pf["tokens_1m"]["baseline_median_s"]
assert r163 > 0.42 and r1m > 0.40, (r163, r1m)

# 4. 10x1M, 24x1M & mixed-load invariants
c10 = DATA["capacity_10x1m"]
assert c10["passed"] and c10["sessions"] == 10 and c10["retrieval_passed"] == 10
assert c10["cold_prefix_hits"] == 0 and c10["waiting_during_hold"] == 0 and c10["preemptions"] == 0
assert c10["hold_duration_s"] >= 60.0 and c10["cleanup_complete"]
assert c10["cold_ttft_max_s"] < c10["baseline_20261114_ttft_min_med_max_s"][2] * 0.65

c24 = DATA["historical_capacity_24x1m_20261109"]
assert c24["passed"] and c24["sessions"] == 24 and c24["retrieval_passed"] == 24
assert c24["preemptions"] == 0 and c24["aggregate_decode_tps"] > 750.0

mx = DATA["mixed_c10x128k_plus_1m"]
assert mx["passed"] and mx["cold_prefix_hits"] == 0
assert mx["cand4_20261122"]["baseline_window_agg_tps"] > mx["baseline_20261109"]["baseline_window_agg_tps"]
assert mx["cand4_20261122"]["competing_prefill_agg_tps"] > mx["baseline_20261109"]["competing_prefill_agg_tps"]
assert mx["cand4_20261122"]["new_million_ttft_s"] < mx["baseline_20261109"]["new_million_ttft_s"]
assert mx["cand4_20261122"]["recovery_window_agg_tps"] > mx["baseline_20261109"]["recovery_window_agg_tps"]

# 5. Quality gates & rejected candidates
q = DATA["qualification"]
assert q["gsm8k_200_1024_budget"]["passed"] == 197 and q["gsm8k_200_1024_budget"]["truncated"] == 0
assert q["quality_chat"]["passed"] == 28 and q["quality_low"]["passed"] == 30
assert q["production_quality_t1"]["passed"] >= q["production_quality_t1"]["gate_min"]
assert q["long_quality"]["passed"] == 12 and q["long_vision"]["passed"] == 4
assert q["application_checks"]["passed"] == 10 and q["short_vision"]["passed"] == 2
assert q["public_https_acceptance"]["passed"] == 10 and q["public_cold_cancel"]["passed"] == 3
assert all(q["post_promotion_three_protocol_check"].values())
assert any(x["name"] == "cand5-ws19-ws23-mxfp8tune" and x["outcome"] == "rejected" for x in DATA["rejected_candidates"])

# 6. Asset existence check
for asset in (
    "assets/deepseek-v41-speed9-hero-20261122.jpg",
    "assets/deepseek-v41-speed9-architecture-20261122.png",
    "assets/deepseek-v41-8gpu-arch-tradeoff-20261124.png",
    "assets/deepseek-v41-8gpu-decode-throughput-20261124.png",
    "assets/deepseek-v41-speed9-prefill-ttft-20261122.png",
):
    assert (ROOT / asset).is_file(), f"Missing asset: {asset}"

# 7. Sanitization & strict 8-GPU-only isolation scan
forbidden_secrets = [
    re.compile(r"\b10\.168\.\d+\.\d+\b"),
    re.compile(r"\b192\.168\.\d+\.\d+\b"),
    re.compile(r"\b64\.64\.236\.12\b"),
    re.compile(r"/opt/", re.I),
    re.compile(r"D:\\server", re.I),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"api\.kingsing\.com", re.I),
]
forbidden_4gpu_mixing = [
    re.compile(r"\bfour[- ]gpu\b", re.I),
    re.compile(r"\bfour GPUs\b", re.I),
    re.compile(r"\b4[- ]GPU\b", re.I),
    re.compile(r"四卡"),
    re.compile(r"\b6[x×]1M\b", re.I),
    re.compile(r"\b11\.00\s*GiB\b", re.I),
    re.compile(r"Speed10", re.I),
    re.compile(r"KINGSING_DeepSeek_V41_Local_AI_Deployment", re.I),
]

check_files = [
    "README.md",
    "docs/eight-gpu-deployment-en.md",
    "docs/eight-gpu-deployment-zh.md",
    "data/2026-11-22-eight-gpu-speed9.json",
    "configs/2026-11-22-eight-gpu-speed9.json",
]
for rel in check_files:
    txt = (ROOT / rel).read_text(encoding="utf-8")
    for pat in forbidden_secrets:
        m = pat.search(txt)
        assert m is None, f"Sanitization violation in {rel}: {m.group(0)}"
    for pat in forbidden_4gpu_mixing:
        m = pat.search(txt)
        assert m is None, f"4-GPU mixing violation in 8-GPU repo ({rel}): {m.group(0)}"

print("PASS: 8-GPU Speed9 invariants, decode/prefill gains, 10x1M/24x1M capacity, quality, sanitization & strict 8-GPU isolation verified.")
