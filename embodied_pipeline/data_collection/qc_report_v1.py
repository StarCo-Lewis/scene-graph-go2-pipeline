"""Assemble audited AI reviews plus evidence hashes; no additional model calls."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import json
import shutil

from .batch import read, read_jsonl, write, jsonl, sha
from .qc_rules_v1 import SCENES

STAGES = {'D': '任务分解', 'B': '物体绑定', 'S': '技能绑定', 'E': '最终工程JSON'}
CODE_GROUPS = {
    'missing_release_goal': 'missing_final_release', 'missing_final_release_predicate': 'missing_final_release',
    'missing_required_final_held': 'missing_final_release',
    'missing_hold_destination_goal': 'missing_final_destination', 'missing_final_destination_predicate': 'missing_final_destination',
    'missing_required_final_near': 'missing_final_destination',
    'intermediate_visits_as_final_goals': 'intermediate_as_final',
    'intermediate_visits_as_conjunctive_final_goals': 'intermediate_as_final',
    'intermediate_visits_encoded_as_final': 'intermediate_as_final',
    'protection_not_repeated_in_task_text': 'protection_text_omission',
    'protection_not_repeated_in_intent_text': 'protection_text_omission',
    'transport_split_into_atomic_phases': 'atomic_overfragmentation',
}


def summarize(records):
    return {'scored_files': len(records),
            'score_distribution': dict(sorted(Counter(r['total_score'] for r in records).items())),
            'mean_total_score': round(sum(r['total_score'] for r in records) / len(records), 4) if records else None,
            'full_score_no_observed_deduction': sum(r['total_score'] == 12 for r in records),
            'critical_error_files': sum(r['critical_error'] for r in records),
            'minor_only_files': sum(bool(r['issues']) and not r['critical_error'] for r in records),
            'end_to_end_critical_files': sum(r['fidelity_scores']['E'] == 0 for r in records),
            'stage_fidelity_distribution': {s: dict(Counter(r['fidelity_scores'][s] for r in records)) for s in STAGES}}


def assemble(root):
    dest = root / 'reports/quality_check_v1'
    automated = {r['record_id']: r for r in read_jsonl(dest / 'automated_checks.jsonl')}
    records, source_manifest, methods = [], {}, {}
    now = datetime.now(timezone.utc).isoformat()

    def source(path):
        path = Path(path)
        relative = path.relative_to(root).as_posix()
        if relative not in source_manifest:
            source_manifest[relative] = {'path': relative, 'sha256': sha(path), 'bytes': path.stat().st_size}
        return relative

    for scene in SCENES:
        review_path = dest / f'reviewer_{scene}.json'
        review = read(review_path)
        methods[scene] = {k: v for k, v in review.items() if k != 'records'}
        candidates = read_jsonl(root / 'cases' / scene / 'tasks.jsonl')
        expected = {t['record_id'] for t in candidates if (root / 'cases' / scene / 'tasks' / t['record_id'] / 'model_run/engineering_plan.json').exists()}
        rows = review['records']
        assert len(rows) == len(expected) and {r['record_id'] for r in rows} == expected, f'Incomplete review: {scene}'
        for reviewed in rows:
            rid = reviewed['record_id']
            assert reviewed['reviewed'] and set(reviewed['scores']) == set(STAGES)
            assert all(type(v) is int and 0 <= v <= 2 for v in reviewed['scores'].values())
            run = root / 'cases' / scene / 'tasks' / rid / 'model_run'
            e = read(run / 'engineering_plan.json')
            d = read(run / 'task_decomposition.json')
            b = read(run / 'object_alignment.json')
            s = read(run / 'skill_plan.json')
            original_hash = sha(run / 'engineering_plan.json')
            assert original_hash == automated[rid]['source_engineering_sha256']
            if reviewed.get('source_engineering_sha256'):
                assert original_hash == reviewed['source_engineering_sha256']
            # Historical stage completion, deliberately NOT new semantic acceptance.
            reliability = {'D': int(bool(d.get('subtasks'))), 'B': int(b.get('status') == 'success'),
                           'S': int(bool(s.get('subtask_plans')) and not s.get('unplanned_subtasks')),
                           'E': int(e.get('task_id') == rid and isinstance(e.get('plan'), dict))}
            assert sum(reliability.values()) == 4, f'Unexpected stage-completion evidence: {rid}'
            issues = reviewed.get('issues', [])
            for issue in issues:
                assert issue['stage'] in STAGES and issue['severity'] in ('minor', 'critical') and issue.get('evidence'), rid
                issue['category'] = CODE_GROUPS.get(issue['code'], issue['code'])
                for evidence in issue['evidence']:
                    if isinstance(evidence, dict) and evidence.get('file'):
                        path = Path(evidence['file'])
                        if not path.is_absolute():
                            path = root / path
                        assert path.is_file(), f'Missing evidence: {path}'
            for stage, value in reviewed['scores'].items():
                stage_issues = [i for i in issues if i['stage'] == stage]
                expected_score = min([0 if i['severity'] == 'critical' else 1 for i in stage_issues], default=2)
                assert expected_score == value, f'Unsupported score: {rid}/{stage} {value}!={expected_score}'
            sources = {}
            for name in ['task.json', 'reference.json']:
                sources[name] = source(run.parent / name)
            for name in ['task_decomposition.json', 'object_alignment.json', 'skill_plan.json', 'engineering_plan.json',
                         'planning_summary.json', 'execution_skill_library.json', 'process_constraints_grounded.json',
                         'llm_decomposition_attempt_1.json', 'llm_binding_proposal.json', 'llm_matrix_proposal.json']:
                if (run / name).is_file():
                    sources[name] = source(run / name)
            for name in ['scene_graph.json', 'scene_graph_edges.json']:
                sources[name] = source(root / 'cases' / scene / 'graph' / name)
            sources['actual_model_requests'] = [source(p) for p in sorted((run / 'llm_response_audit').glob('*_request.json'))]
            sources['raw_http_responses'] = [source(p) for p in sorted((run / 'llm_response_audit').glob('*_http_*.json'))]
            total = sum(reliability.values()) + sum(reviewed['scores'].values())
            critical = any(i['severity'] == 'critical' for i in issues)
            record = {'record_id': rid, 'scene_id': scene, 'instruction': e['instruction'],
                      'rubric_version': 'pipeline_static_12point_v1', 'reviewed_at_utc': now,
                      'review_type': 'AI_static_semantic_review_with_reproducible_checks',
                      'independent_human_review': False, 'physical_feasibility_verified': False,
                      'robot_executed_by_review': False, 'original_engineering_sha256': original_hash,
                      'reliability_scores': reliability, 'reliability_total': sum(reliability.values()),
                      'fidelity_scores': reviewed['scores'], 'fidelity_total': sum(reviewed['scores'].values()),
                      'total_score': total, 'maximum_score': 12, 'critical_error': critical,
                      'review_status': 'needs_correction' if critical else 'minor_deviation' if issues else 'no_observed_deduction_static_only',
                      'issues': issues, 'notes': reviewed.get('notes', []), 'evidence_sources': sources,
                      'evidence_rendering': 'Some observations resolve role IDs to scene IDs for readability; original JSON plus binding file are authoritative.',
                      'source_reference_base': 'original bundle root', 'not_robot_execution_authorization': True}
            records.append(record)
    records.sort(key=lambda r: (SCENES.index(r['scene_id']), r['record_id']))
    counts = defaultdict(set)
    for r in records:
        for i in r['issues']:
            if i['stage'] != 'E' or i.get('origin') not in ('propagated', 'inherited'):
                counts[i['category']].add(r['record_id'])
    report = {'version': 'pipeline_static_quality_report_v1', 'created_at_utc': now, 'selected_scenes': list(SCENES),
              'source_bundle_root': str(root), 'collection_records': 600, 'scored_engineering_files': len(records),
              'not_generated_unscored_records': len(read_jsonl(dest / 'not_generated.jsonl')),
              'summary': summarize(records), 'per_scene': {scene: summarize([r for r in records if r['scene_id'] == scene]) for scene in SCENES},
              'issue_categories_unique_record_counts': {k: len(v) for k, v in counts.items()},
              'issue_categories_overlap': True, 'original_model': 'qwen3.8-flash',
              'reviewer': 'Current Codex session and collaborating AI reviewers; exact serving model revision unavailable',
              'external_api_calls_by_quality_check': 0, 'independent_human_review': False,
              'source_graph_physical_truth_verified': False, 'semantic_reference_human_verified': False,
              'training_eligibility_not_granted': True, 'methods': methods,
              'interpretation': ['Full score means no observed deduction under this static rubric, not proven real-world success.',
                  'All 562 scored files completed generation, so stage-completion scores are 4/4 by construction; do not infer all-600 completion from this subset.',
                  'Missing structured release/destination conditions are scored in decomposition; correct downstream effects can recover final behavior.',
                  'Unsupported initial conditions or conjunctive intermediate final goals retained in engineering JSON are marked propagated end-to-end errors.',
                  'References are authored task specifications, not independently verified oracles; alternate equivalent plans are accepted.',
                  'No successful score asserts real navigation/graspability, scene perception truth, or safety.']}
    write(dest / 'summary.json', report)
    jsonl(dest / 'scores.jsonl', records)
    jsonl(dest / 'errors.jsonl', [r for r in records if r['issues']])
    jsonl(dest / 'source_evidence_manifest.jsonl', sorted(source_manifest.values(), key=lambda x: x['path']))
    for scene in SCENES:
        selected = [r for r in records if r['scene_id'] == scene]
        jsonl(dest / scene / 'scores.jsonl', selected)
        jsonl(dest / scene / 'errors.jsonl', [r for r in selected if r['issues']])
        for record in selected:
            write(dest / scene / (record['record_id'] + '.quality.json'), record)
    rubric = {'version': 'pipeline_static_12point_v1', 'stages': STAGES,
              'reliability': {'max': 4, 'criterion': 'Historical stage completion 0/1, not semantic correctness.'},
              'fidelity': {'max': 8, '2': 'No observed conflict with supported instructions/facts/contracts.',
                           '1': 'Noncritical textual or organizational deviation; original actual action plan need not change.',
                           '0': 'Material fact/goal/constraint/protocol/action error.', 'NA': 'No evaluable output; do not fabricate a total.'},
              'critical_error_overrides_total': True, 'scope': '562 generated engineering JSONs; 38 missing files separately listed, not scored as complete files.',
              'equivalence_rules': ['Held payload near destination is accepted as carry endpoint; robot-subject near is not mandatory.',
                                    'Placement skill releases by definition; lack of held=false in decomposition does not itself make a correct placement sequence wrong.',
                                    'Navigation to the supported source can substitute approaching the payload itself.',
                                    'Unknown runtime checks left unknown and execution_ready=false are not errors.'],
              'non_scored_engineering_risks': ['Protected-object process restrictions are stored externally and not consistently embedded in engineering plan; downstream enforcement requires separate interface audit.']}
    write(dest / 'rubric.json', rubric)
    for script in [Path(__file__), Path(__file__).with_name('qc_rules_v1.py')]:
        shutil.copy2(script, dest / script.name)
    render_html(dest, records, report)
    return report


def render_html(dest, records, report):
    def h(value):
        return escape(str(value))
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>三场景工程 JSON 质量审阅</title>',
             '<style>body{font:15px/1.6 system-ui,sans-serif;max-width:1500px;margin:36px auto;padding:0 24px;color:#16283a}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #cad3dc;padding:8px;text-align:left;vertical-align:top}th{background:#eaf1f7;position:sticky;top:0}pre{white-space:pre-wrap;word-break:break-word;background:#f2f5f8;padding:12px}details{margin:8px 0}.bad{color:#a32824}.minor{color:#885909}h1,h2{color:#183b59}summary{cursor:pointer}small{color:#506477}</style>',
             '<h1>Scene200 / Scene202 / D3-619 工程 JSON 质量审阅</h1>',
             '<p>范围：600 条采集记录中的 562 份工程 JSON。另 38 条未生成工程 JSON，单独保留，不填造完整评分。每份按流程完成度 4 分＋忠实性 8 分评分。</p>',
             '<p><b>这是 AI 静态语义审阅与规则核验，不是独立人工审核、图像真值核验或机器人实测。</b>12 分仅表示本轮未发现扣分项。原文件未修改。</p>',
             '<p>D＝任务分解；B＝物体绑定；S＝技能绑定；E＝最终工程 JSON。关键错误不会被高总分抵消；“传播”不算新的独立根因。</p>',
             '<table><tr><th>场景</th><th>已评文件</th><th>12分</th><th>仅轻微偏差</th><th>有关键错误</th><th>平均分</th></tr>']
    for scene, s in report['per_scene'].items():
        parts.append('<tr>' + ''.join(f'<td>{h(v)}</td>' for v in [scene,s['scored_files'],s['full_score_no_observed_deduction'],s['minor_only_files'],s['critical_error_files'],s['mean_total_score']]) + '</tr>')
    parts.extend(['</table><p>以下每行可展开查看任务、扣分原因和字段证据。Ctrl+F 可按任务编号检索。</p>',
                  '<table><tr><th>任务编号</th><th>完成度</th><th>忠实性 D/B/S/E</th><th>总分</th><th>问题与证据</th></tr>'])
    for r in records:
        parts.append(f'<tr><td>{h(r["record_id"])}</td><td>{r["reliability_total"]}/4</td><td>{" / ".join(str(r["fidelity_scores"][s]) for s in STAGES)}</td><td>{r["total_score"]}/12</td><td>')
        cls = 'bad' if r['critical_error'] else 'minor' if r['issues'] else ''
        label = '存在关键错误' if r['critical_error'] else '存在轻微偏差' if r['issues'] else '本轮未发现扣分'
        parts.append(f'<details><summary class="{cls}">{label} · {len(r["issues"])} 项</summary><p>{h(r["instruction"])}</p>')
        for i in r['issues']:
            parts.append(f'<p><b>{h(STAGES[i["stage"]])} / {h(i["severity"])} / {h(i.get("origin", "undetermined"))}</b>：{h(i["reason"])}</p>')
            parts.append('<details><summary>字段证据</summary><pre>' + h(json.dumps(i['evidence'],ensure_ascii=False,indent=2)) + '</pre></details>')
        for note in r['notes']:
            parts.append('<p><small>' + h(note) + '</small></p>')
        parts.append(f'<p><a href="{h(r["scene_id"] + "/" + r["record_id"] + ".quality.json")}">逐条评分 JSON</a></p></details></td></tr>')
    parts.append('</table><p>关联源文件以原数据包根目录为基准。不要将评分侧文件解释为执行授权或人工验证。</p></html>')
    (dest / 'review_report.html').write_text('\n'.join(parts), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    result = assemble(p.parse_args().bundle)
    print(json.dumps({'summary': result['summary'], 'per_scene': result['per_scene']}, ensure_ascii=False))
