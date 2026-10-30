#!/usr/bin/env python3
"""Summarize GNU time and in-process Habitat memory measurements."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def meminfo():
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                values[key] = round(int(raw.split()[0]) / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-file", type=Path, required=True)
    parser.add_argument("--proxy-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.time_file.read_text(encoding="utf-8", errors="replace") if args.time_file.is_file() else ""
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    time_peak_mb = round(int(match.group(1)) / 1024, 1) if match else None
    proxy = json.loads(args.proxy_json.read_text(encoding="utf-8"))
    internal_peak = proxy.get("resource_summary", {}).get("process_peak_rss_mb")
    peaks = [x for x in (time_peak_mb, internal_peak) if isinstance(x, (int, float))]
    peak_mb = max(peaks) if peaks else None
    system = meminfo()
    total = system.get("MemTotal")
    fraction = round(peak_mb / total, 4) if peak_mb is not None and total else None
    successful_statuses = {"proxy_instance_support_passed", "spot_arm_smoke_passed",
                           "spot_ik_reachability_passed",
                           "spot_approach_collision_check_passed",
                           "spot_safe_base_pose_found",
                           "spot_grasp_contact_candidate_found",
                           "spot_grasp_target_candidate_found",
                           "spot_dynamic_lift_validated"}
    if proxy.get("status") not in successful_statuses:
        recommendation = "benchmark_failed_no_capacity_decision"
    elif fraction is None:
        recommendation = "local_run_passed_but_peak_ratio_unavailable"
    elif fraction < 0.6:
        recommendation = "local_machine_adequate_for_current_headless_proxy"
    elif fraction < 0.8:
        recommendation = "local_machine_borderline_for_current_headless_proxy"
    else:
        recommendation = "more_memory_recommended_for_current_headless_proxy"
    result = {
        "schema_version": "1.0",
        "status": "measured" if peak_mb is not None else "partial",
        "episode_status": proxy.get("status"),
        "peak_rss_mb": peak_mb,
        "peak_rss_gib": round(peak_mb / 1024, 3) if peak_mb is not None else None,
        "gnu_time_peak_rss_mb": time_peak_mb,
        "internal_vmhwm_mb": internal_peak,
        "wsl_memory_total_mb": total,
        "wsl_memory_available_after_run_mb": system.get("MemAvailable"),
        "peak_fraction_of_wsl_total": fraction,
        "recommendation": recommendation,
        "scope": "Current headless episode only; rendering, larger scenes, multiple sensors, and training may require more memory.",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
