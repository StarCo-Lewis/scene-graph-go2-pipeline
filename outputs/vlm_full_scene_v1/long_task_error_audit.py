"""Read stored rollout evidence after all dialogues; no model calls or plan repair."""
import json
from pathlib import Path
from run_navigation_demo import write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def category(error,status):
    if status=='proposal_unavailable':
        return 'decomposition_binding_or_wire_protocol_review_required'
    if 'GRIPPER_NOT_EMPTY' in error:
        return 'occupied_gripper_prerequisite_missing'
    if 'SURFACE_PROXY_PREPARATION_REQUIRED' in error:
        return 'placement_surface_preparation_missing_or_wrong_target'
    if 'PROXY_PREPARATION_REQUIRED' in error:
        return 'grasp_preparation_missing_or_wrong_target'
    if 'UNSUPPORTED_SYMBOLIC' in error:
        return 'fixture_capability_boundary_or_selection_review_required'
    if error:
        return 'proxy_execution_precondition_or_action_error'
    return None


def build(root):
    root=Path(root)
    batch=read(root/'dialogue_summary.json')
    rows=[]
    for ep in batch['episodes']:
        folder=root/ep['path']
        audit_path=folder/'evaluation/final_audit.json'
        audits=read(audit_path)['rows'] if audit_path.is_file() else []
        indexed={a['round']:a for a in audits}
        for turn in ep['rounds']:
            planning=folder/turn['planning_path']
            diagnostic=indexed.get(turn['round'],{})
            execution=folder/f"round_{turn['round']}/execution_feedback.json"
            feedback=read(execution) if execution.is_file() else {}
            error=turn.get('error')
            status=turn['status']
            if ep['status']=='harness_error' and 'output_memory_version' not in turn:
                status='rollout_infrastructure_error'
                error=ep.get('error','Infrastructure/integrity error; no state commit')
            failed=[k for k,v in diagnostic.get('reference_checks',{}).items() if v is False]
            strict=diagnostic.get('strict_id_visit_sequence_matched')
            ordinary_failure=turn['status'] in ('proxy_action_failed','proposal_unavailable')
            deviation=ordinary_failure or bool(failed) or strict is False or status=='rollout_infrastructure_error'
            label=('infrastructure_or_integrity_not_model_action_error' if status=='rollout_infrastructure_error'
                   else category(error or '',status))
            if label is None and failed: label='post_rollout_reference_mismatch_manual_semantic_review_required'
            if label is None and strict is False: label='strict_visit_id_or_payload_mismatch_check_geometry_before_blame'
            paths={key:str(path.relative_to(root)) for key,path in {
                'input':planning/'round_input.json','state_before':planning/'runtime_state.json',
                'decomposition':planning/'task_decomposition.json',
                'raw_bindings':planning/('llm_binding_matrix_proposal_attempt_1.json'
                    if (planning/'object_binding_codebook.json').is_file() else 'llm_binding_proposal.json'),
                'decoded_bindings':planning/'llm_binding_decoded_attempt_1.json',
                'binding_codebook':planning/'object_binding_codebook.json',
                'raw_decomposition':planning/'llm_decomposition_attempt_1.json',
                'decomposition_codebook':planning/'decomposition_codebook_attempt_1.json',
                'effective_alignment':planning/'object_alignment.json',
                'raw_matrix':planning/'llm_matrix_proposal_semantic_1_attempt_1.json',
                'codebook':planning/'action_codebook_semantic_1.json',
                'proposal':planning/'skill_plan_proposed.json','feedback':execution,
                'state_after':planning.parent/'memory_after_feedback.json',
                'diagnostic':planning.parent/'evaluation/post_rollout_diagnostics.json',
                'planning_summary':planning/'planning_summary.json'}.items() if path.is_file()}
            rows.append(dict(episode_id=ep['episode_id'],scene=ep['scene'],repeat=ep['repeat'],
                user_location=ep.get('user_location'),
                round=turn['round'],instruction=turn['instruction'],status=status,
                error=error,failure=feedback.get('failure'),post_rollout_audit_available=bool(diagnostic),
                observed_deviation=deviation,diagnostic_category=label,
                reference_mismatches=failed,strict_id_visits_matched=strict,
                dependency_reference_matched=diagnostic.get('dependency_reference_matched'),
                classification_scope='Candidate localization, not proof of LLM internal cause; inspect raw evidence.',
                evidence=paths))
    result=dict(batch_status=batch['status'],planned_round_count=batch['planned_round_count'],
        processed_round_count=batch['processed_round_count'],rows=rows,
        semantic_success_rate=None,physical_execution=False,
        first_observed_deviations=[next((r for r in rows if r['episode_id']==e['episode_id']
            and r['repeat']==e['repeat'] and r['observed_deviation']),None) for e in batch['episodes']],
        scope='Post-rollout evidence only. Failed dependencies may cause propagation, not independent model errors.')
    write_json(root/'evaluation/error_localization.json',result)
    lines=['# 十任务长程对话：运行后错误定位','',
        f"批次状态：`{batch['status']}`；处理轮数：{batch['processed_round_count']} / {batch['planned_round_count']}。",'',
        '这不是语义成功率报告；预检不计成功。首次异常是首次可观测代理/参考偏离，不保证等于首次 LLM 语义错误。',
        '候选归因需核对原始分解、绑定、矩阵及执行轨迹；前序目标未实现时，后续参考偏离可能是传播或前提失效。',
        '共址表面与物体的严格 ID 差异须复核几何含义；瓶内有水、可饮用和解渴不能由搬运成功推出。','',
        '| 任务 | 场景 / 用户位置 | 轮次 | 代理状态 | 异常/待复核项目 | 证据 |','|---|---|---:|---|---|---|']
    for row in rows:
        explanation=row['error'] or ', '.join(row['reference_mismatches']) or (
            '严格访问 ID/携带对象待复核' if row['strict_id_visits_matched'] is False else '未见自动检查异常；语义仍待审计')
        if row['status']=='first_round_input_preflight_passed': explanation='仅输入预检，未运行模型'
        if row['dependency_reference_matched'] is False: explanation+='；输入已偏离预期历史，需区分传播与新错'
        evidence=row['evidence'].get('feedback',row['evidence'].get('input',''))
        explanation=explanation.replace('|','/').replace('\n',' ')
        room=(row['user_location'] or {}).get('room_name','unknown')
        lines.append(f"| {row['episode_id']} r{row['repeat']} | {row['scene']} / {room} | {row['round']} | {row['status']} | {explanation} | [记录](../{evidence.replace(chr(92),'/')}) |")
    lines.extend(['','详细分类、失败动作编号、技能、缺失检查与全部相对证据路径见 `error_localization.json`。',
        '完整链处理完、代理动作未报错、参考约束匹配与真实语义成功是不同指标；本批没有物理执行。',''])
    (root/'evaluation/错误定位.md').write_text('\n'.join(lines),encoding='utf-8')
    return result
