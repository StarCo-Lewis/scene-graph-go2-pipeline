"""Offline regression of the semantic gate against the 30 frozen pilot runs.

Reads the saved artifacts only. No model call, no WSL, no robot, and no write
inside the two batch directories; every output goes to a separate report
directory. The run fails if any source file hash changes while it reads them.
"""

import hashlib
import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embodied_pipeline.validation import plan_semantics  # noqa: E402

BATCHES = [ROOT / "outputs/symbolic_pilot_12_20261031_ready",
           ROOT / "outputs/symbolic_extension18_20261031_ready_v2"]
REVIEW = ROOT / "outputs/symbolic30_review_20261031_v1/review.json"
OUT = ROOT / "outputs/symbolic30_audit_regression_20261031_v1"
GATE_STAGE = "skill_matching_matrix"

# Frozen manual labels mapped to the issue the gate must raise for them.
EXPECTED = {
    "terminal_goal_error": "mutually_exclusive_terminal_robot_location",
    "clarification_with_decomposition_issue": "mutually_exclusive_terminal_robot_location",
    "protection_gap": "preservation_requirement_not_structured",
    "pass": None,
    "expected_clarification": None,
}


def longpath(path):
    """Windows refuses paths over 260 characters without this prefix."""
    path = Path(path).resolve()
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(path))
    return path


def read(path):
    return json.loads(longpath(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with longpath(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory():
    """Hash every saved artifact so the regression can prove it changed nothing."""
    return {str(p.relative_to(ROOT)): sha(p)
            for base in BATCHES for p in sorted((base / "results").rglob("*")) if p.is_file()}


def gate_stage_usage(folder):
    """Tokens the blocked skill-binding request actually consumed in the frozen run."""
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0}
    directory = folder / "llm_response_audit"
    if not directory.is_dir():
        return usage
    for path in sorted(directory.iterdir()):
        if "_http_" not in path.name or not path.name.startswith(GATE_STAGE):
            continue
        body = json.loads(read(path)["response_body"])
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] += body["usage"][key]
        usage["requests"] += 1
    return usage


def audit_record(folder, instruction):
    """Replay the gate where the pipeline runs it, then again with the plan."""
    decomposition = read(folder / "task_decomposition.json")
    alignment = read(folder / "object_alignment.json")
    bindings = alignment.get("context_bindings") if alignment.get("status") == "success" else None
    gate = plan_semantics.audit(decomposition, bindings=bindings, instruction=instruction)
    plan_path = folder / "engineering_plan.json"
    with_plan = None
    if plan_path.is_file():
        with_plan = plan_semantics.audit(decomposition, bindings=bindings,
                                         plan=read(plan_path)["plan"], instruction=instruction)
    return gate, with_plan, alignment


def main():
    before = inventory()
    labels = {r["record_id"]: r["review_result"] for r in read(REVIEW)["rows"]}
    rows, avoided = [], {"tasks": 0, "requests": 0, "prompt_tokens": 0,
                         "completion_tokens": 0, "total_tokens": 0}
    for base in BATCHES:
        for folder in sorted((base / "results").iterdir()):
            if not folder.is_dir():
                continue
            record = folder.name
            instruction = read(folder / "task_input.json")["instruction"]
            gate, with_plan, alignment = audit_record(folder, instruction)
            expected = EXPECTED[labels[record]]
            detected = gate["defect_codes"]
            correct = (expected in detected) if expected else not detected
            if gate["has_defect"]:
                usage = gate_stage_usage(folder)
                avoided["tasks"] += 1
                for key in ("requests", "prompt_tokens", "completion_tokens", "total_tokens"):
                    avoided[key] += usage[key]
            rows.append({
                "record_id": record, "batch": base.name, "instruction": instruction,
                "frozen_review_label": labels[record], "expected_gate_issue": expected,
                "gate_flags_defect": gate["has_defect"], "gate_issue_codes": gate["issue_codes"],
                "gate_detail": [i["detail"] for i in gate["issues"]],
                "suggested_terminal_goal": gate.get("suggested_terminal_goal"),
                "agrees_with_frozen_review": correct,
                "object_binding_status": alignment.get("status"),
                "binding_error_code": (alignment.get("error") or {}).get("code"),
                "engineering_plan_present": with_plan is not None,
                "post_plan_issue_codes": with_plan["issue_codes"] if with_plan else None,
                "post_plan_adds_issue": bool(with_plan) and bool(
                    set(with_plan["issue_codes"]) - set(gate["issue_codes"])),
            })
    expected_hits = [r for r in rows if r["expected_gate_issue"]]
    clean = [r for r in rows if not r["expected_gate_issue"]]
    stats = {
        "regression_version": "symbolic30_semantic_gate_regression_v1",
        "audit_version": plan_semantics.VERSION,
        "new_api_calls": 0, "robot_executed": False,
        "records": len(rows),
        "known_defects": len(expected_hits),
        "known_defects_detected": sum(1 for r in expected_hits if r["agrees_with_frozen_review"]),
        "missed_defects": [r["record_id"] for r in expected_hits if not r["agrees_with_frozen_review"]],
        "clean_records": len(clean),
        "false_positives": [r["record_id"] for r in clean if r["gate_flags_defect"]],
        "detected_by_code": {},
        "flagged_records_skill_binding_usage": avoided,
        "engineering_json_delivered": sum(1 for r in rows if r["engineering_plan_present"]),
        "delivery_policy": ("annotate: every task keeps its engineering JSON; the audit only "
                            "stamps quality_flags for downstream filtering."),
        "post_plan_only_findings": [r["record_id"] for r in rows if r["post_plan_adds_issue"]],
        "scope": ("Replay of frozen outputs against deterministic invariants. It shows the gate "
                  "reproduces the known defects on this purposive sample; it is not a new accuracy "
                  "measurement, a guarantee for unseen error modes, or robot validation."),
        "rows": rows,
    }
    for row in rows:
        for code in row["gate_issue_codes"]:
            stats["detected_by_code"][code] = stats["detected_by_code"].get(code, 0) + 1
    after = inventory()
    changed = sorted(k for k in before if before[k] != after.get(k))
    stats["source_files_checked"] = len(before)
    stats["source_files_changed"] = changed
    if changed or set(before) != set(after):
        raise SystemExit("Frozen batch artifacts changed during the regression: " + str(changed[:5]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "regression.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    (OUT / "source_inventory_sha256.json").write_text(
        json.dumps(before, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_html(stats)
    print(json.dumps({k: stats[k] for k in (
        "records", "known_defects", "known_defects_detected", "missed_defects",
        "clean_records", "false_positives", "detected_by_code",
        "flagged_records_skill_binding_usage", "engineering_json_delivered",
        "post_plan_only_findings", "source_files_checked")},
        ensure_ascii=False, indent=2))


def write_html(stats):
    e = html.escape
    labels = {"pass": "符合参考", "terminal_goal_error": "终态错误", "protection_gap": "保护约束缺口",
              "expected_clarification": "合理澄清／拒绝",
              "clarification_with_decomposition_issue": "合理澄清，但分解有错"}
    body = ""
    for r in stats["rows"]:
        verdict = "标注缺陷" if r["gate_flags_defect"] else "通过"
        ok = "一致" if r["agrees_with_frozen_review"] else "不一致"
        body += ("<tr><td>" + e(r["record_id"]) + "</td><td>" + e(r["instruction"]) + "</td><td>"
                 + labels[r["frozen_review_label"]] + "</td><td>" + verdict + "</td><td>"
                 + e(", ".join(r["gate_issue_codes"]) or "-") + "</td><td>" + ok + "</td></tr>")
    summary = {k: stats[k] for k in (
        "known_defects", "known_defects_detected", "missed_defects", "clean_records",
        "false_positives", "detected_by_code", "flagged_records_skill_binding_usage",
        "engineering_json_delivered", "source_files_checked", "source_files_changed")}
    doc = ("<!doctype html><html lang=\"zh\"><meta charset=\"utf-8\"><title>语义闸门离线回归</title>"
           "<style>body{font:16px/1.65 system-ui;max-width:1300px;margin:35px auto;padding:20px;"
           "color:#183047}table{width:100%;border-collapse:collapse}td,th{border:1px solid #ccd6df;"
           "padding:9px;vertical-align:top;text-align:left}th{background:#eaf2f8}pre{white-space:"
           "pre-wrap;background:#f3f5f8;padding:16px}</style>"
           "<h1>语义闸门对 30 条冻结结果的离线回归</h1>"
           "<p>只读回放已保存的分解、绑定与工程 JSON；无模型调用、无 WSL、无真机，"
           "原始文件哈希在回归前后一致。默认策略是全量产出：每条任务都保留工程 JSON，"
           "检查只写 quality_flags 供后续筛选。本页说明新检查能否复现已知缺陷，"
           "不是新的准确率，也不保证覆盖未见过的错误类型。</p>"
           "<pre>" + e(json.dumps(summary, ensure_ascii=False, indent=2)) + "</pre>"
           "<table><tr><th>记录</th><th>指令</th><th>冻结人工标签</th><th>闸门判定</th>"
           "<th>命中检查项</th><th>与标签</th></tr>" + body + "</table></html>")
    (OUT / "report.html").write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    main()
