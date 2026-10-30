"""Aggregate new audit records only; never guess old missing LLM responses."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from response_audit import save


def build(run_dir, output_dir):
    root, destination = Path(run_dir), Path(output_dir)
    records = []
    for path in sorted(root.rglob("llm_response_audit/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        records.append((path, data))
    transport = [(p,r) for p,r in records if r.get("event") in ("http_response","transport_failure")]
    http_errors = [(p,r) for p,r in transport if (r.get('http_status') or 0) >= 400]
    validation = [(p,r) for p,r in records if r.get("event") == "program_validation"]
    rejected = [(p,r) for p,r in validation if r.get("status") == "rejected"]
    # Malformed content is rejected before the program validator gets a parsed proposal.
    parse_failures = [(p,r) for p,r in records if r.get("event") == "content_parse" and r.get("status") == "rejected"]
    request_records = [(p,r) for p,r in records if "model_input" in r]
    rows = [{"file": str(p.relative_to(root)), "stage": r["stage"], "status": r.get("status", "transport_failure"),
             "category": r.get("category"), "error": r.get("error"), "context": r.get("context", {})}
            for p,r in [*rejected,*parse_failures,*[(p,r) for p,r in transport if r["event"]=="transport_failure"]]]
    rows.extend({'file':str(p.relative_to(root)), 'stage':r['stage'], 'status':'http_error',
                 'category':'http_error', 'error':f"HTTP {r['http_status']}; inspect saved response_body",
                 'context':r.get('context',{})} for p,r in http_errors)
    data = {"schema_version":"1.0", "source_run":str(root), "training_eligible":False,
            "record_count":len(records), "logical_request_count":len(request_records),
            "http_attempt_count":len(transport),
            "http_response_count":sum(r["event"]=="http_response" for _,r in transport),
            "transport_failure_count":sum(r["event"]=="transport_failure" for _,r in transport),
            "http_error_count":len(http_errors),
            "requests_without_transport_result_count":len({r['request_id_local'] for _,r in request_records}
                - {r['request_id_local'] for _,r in transport}),
            "program_validation_count":len(validation), "program_rejection_count":len(rejected),
            "content_parse_rejection_count":len(parse_failures),
            "program_rejections_by_stage":dict(Counter(r["stage"] for _,r in rejected)),
            "program_rejections_by_category":dict(Counter(r["category"] for _,r in rejected)),
            "failure_records":rows,
            "scope":"Observed request/validation counts, not provider billing, independent task counts or model success rates; heuristic categories"}
    save(destination/"llm_error_summary.json", data)
    text = ["# LLM 返回值审计报告", "", f"来源：`{root}`", "",
            f"已记录逻辑请求：{len(request_records)}；HTTP 尝试：{len(transport)}；传输失败：{data['transport_failure_count']}；HTTP 错误响应：{len(http_errors)}。",
            f"程序校验：{len(validation)} 次；拒绝：{len(rejected)} 次；内容解析拒绝：{len(parse_failures)} 次。", "",
            "## 错误记录", "", "| 阶段 | 类别 | 错误 | 文件 |", "|---|---|---|---|"]
    for r in rows:
        esc = lambda value: str(value or "").replace("|","\\|").replace("\n"," ")
        text.append("| "+" | ".join(esc(r[k]) for k in ("stage","category","error","file"))+" |")
    if not rows:
        text.append("| — | — | 尚未记录失败，不代表模型没有问题 | — |")
    text += ["", "## 统计边界", "",
             "HTTP 尝试、解析失败和程序校验是不同口径，不能相加当作失败任务数。相同输入的重试不算独立任务。",
             "缺物体、缺边、初始条件不满足可能是输入条件问题，不能直接全部归因于 LLM。分类为关键词启发式，需人工核对原始返回及输入。",
             "保留的是本地原始响应/模型内容及程序拒绝原因；只对凭证做脱敏。没有返回内容的超时记录 response_received=false，不伪造模型回复。",
             "不记录认证头或密钥。报告不是独立语义真值验证，不自动将样本标为可训练数据。旧运行未保存的原始返回不能从日志恢复。", ""]
    destination.mkdir(parents=True, exist_ok=True)
    (destination/"LLM返回值审计报告.md").write_text("\n".join(text), encoding="utf-8")
    return data


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args(); result=build(a.run_dir,a.output_dir)
    print(f"report={a.output_dir / 'LLM返回值审计报告.md'} records={result['record_count']}")
