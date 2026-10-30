#!/usr/bin/env python3
"""Fresh three-instruction pipeline with checkpointed memory and optional RGB-D overlay."""
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from update_memory import read, write, digest

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
SCENE = Path('/mnt/d/NebulaAI/output/batches/scene201_vith_full_20260915_181750/seq_0001__hf_house_201')


def execute(command, log):
    with log.open('w', encoding='utf-8') as stream:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace')
        for line in proc.stdout:
            print(line, end='', flush=True)
            stream.write(line)
            stream.flush()
        code = proc.wait()
    if code:
        raise ValueError(f'Stage failed exit_code={code}; log={log}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--planning-python', default='/home/lewis/miniforge3/envs/conceptgraph/bin/python')
    p.add_argument('--scene-dir', type=Path, default=SCENE / 'vlm_scenegraph_regression/scene201_verified_20260921_113316/scene')
    p.add_argument('--matching-mode', choices=('llm_graph', 'clip'), default='llm_graph')
    p.add_argument('--with-vision', action='store_true')
    p.add_argument('--output-dir', type=Path, default=SCENE / 'full_multiround_memory' / datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f'))
    args = p.parse_args()
    if not os.environ.get('DASHSCOPE_API_KEY'):
        p.error('DASHSCOPE_API_KEY required; use shell launcher')
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'running', 'rounds': [], 'fresh_llm_each_round': True,
        'matching_mode': args.matching_mode,
        'visual_requested': args.with_vision, 'training_eligible': False,
        'scope': 'Fresh planning each instruction; checkpoint-restored proxy execution, not one uninterrupted simulator',
        'base_scene': str(args.scene_dir), 'limitations': ['Assumed grasp',
            'Only the supplied active graph can be used; map completeness is unknown',
            'Follow-up starts are sampled independently; robot motion is not continuous across rounds',
            'Original maps never overwritten']}
    if args.with_vision:
        summary['limitations'].append('Optional visual overlay uses oracle instance labels, not VLM')
    current_scene = args.scene_dir
    memory = None
    stage = 'preflight'
    try:
        base_hash = digest([read(current_scene / 'scene_graph.json'), read(current_scene / 'scene_graph_edges.json')])
        instructions = [('Move the electric kettle to the sideboard.', 67),
                        ('Go to the electric kettle.', 67), ('Go to the sideboard.', 13)]
        for index, (instruction, target_id) in enumerate(instructions, 1):
            print(f'START round={index} instruction={instruction}', flush=True)
            round_dir = root / f'round_{index}'
            round_dir.mkdir()
            planning = round_dir / 'planning'
            command = [sys.executable, str(OUT / 'habitat_demo_v1/run_sideboard_full_pipeline.py'),
                '--planning-python', args.planning_python, '--habitat-python', sys.executable,
                '--scene-dir', str(current_scene), '--instruction', instruction,
                '--matching-mode', args.matching_mode,
                '--expected-target-id', str(target_id), '--planning-only', '--output-dir', str(planning)]
            if index == 1:
                command += ['--adapter-mode', 'kettle_to_sideboard']
            elif target_id == 67:
                command += ['--memory-state', str(memory / 'memory_state.json')]
            stage = f'round_{index}_fresh_planning'
            execute(command, round_dir / 'planning.console.log')
            stage = f'round_{index}_execution'
            if index == 1:
                execution = round_dir / 'execution'
                execution.mkdir()
                assets = SCENE / 'habitat_split_assets_v2'
                memory = round_dir / 'scene_memory'
                execute([sys.executable, str(OUT / 'habitat_demo_v1/run_manipulation_proxy.py'),
                    '--alignment-json', str(planning / 'object_alignment.json'), '--skill-plan-json', str(planning / 'skill_plan_final.json'),
                    '--scene-graph-json', str(current_scene / 'scene_graph.json'),
                    '--stage-config', str(assets / 'scene201_without_kettle.stage_config.json'),
                    '--object-config', str(assets / 'kettle_dynamic.object_config.json'),
                    '--sideboard-config', str(assets / 'sideboard_static.object_config.json'),
                    '--navmesh', str(SCENE / 'scene_package/hf_house_201/hf_house_201.navmesh'),
                    '--output', str(execution / 'manipulation_proxy.json'), '--log-file', str(execution / 'execution.log'),
                    '--assume-grasp-after-approach', '--scene-memory-output-dir', str(memory)], round_dir / 'execution.console.log')
                execution_file = execution / 'manipulation_proxy.json'
                followup = None
            else:
                # Sideboard is unchanged, so use the generic planner without its moved-target test.
                # Bind the result to this exact checkpoint before execution.
                if target_id == 13:
                    target = read(planning / 'planning_target.json')
                    target['memory_source'] = str(memory / 'memory_state.json')
                    write(planning / 'planning_target.json', target)
                    plan_summary = read(planning / 'pipeline_summary.json')
                    plan_summary['memory_target_check'] = {'memory_version': read(memory / 'memory_state.json')['version'],
                        'scope': 'Same immutable checkpoint; sideboard did not move'}
                    write(planning / 'pipeline_summary.json', plan_summary)
                execution = round_dir / 'execution'
                execute([sys.executable, str(HERE / 'run_memory_followup_navigation.py'),
                    '--plan-run', str(planning), '--instruction', instruction, '--expected-target-id', str(target_id),
                    '--start-policy', 'sampled_far', '--start-seed', str(200+index),
                    '--output-dir', str(execution)], round_dir / 'execution.console.log')
                execution_file = execution / 'followup_execution.json'
                followup = execution_file
            current_scene = memory / 'scene'
            execution_result = read(execution_file)
            expected_status = 'proxy_instance_support_passed' if index == 1 else 'followup_proxy_passed'
            if execution_result.get('status') != expected_status:
                raise ValueError(f'Unexpected execution result: {execution_result.get("status")}')
            skill_plan = read(planning / 'skill_plan_final.json')
            summary['rounds'].append({'round': index, 'instruction': instruction, 'status': 'executed',
                'planning_run': str(planning), 'execution': str(execution_file), 'memory': str(memory),
                'action_count': len([s for p in skill_plan['subtask_plans'] for s in p['steps']]),
                'subtask_action_mapping': skill_plan.get('subtask_action_mapping'),
                'execution_status': execution_result.get('status')})
            visual_report = None
            if args.with_vision:
                stage = f'round_{index}_visual_fusion'
                visual_dir = round_dir / 'visual'
                visual_command = [sys.executable, str(HERE / 'capture_visual_memory.py'),
                    '--memory-dir', str(memory), '--output-dir', str(visual_dir)]
                if followup:
                    visual_command += ['--followup-execution', str(followup)]
                execute(visual_command, round_dir / 'visual.console.log')
                visual_report = read(visual_dir / 'visual_summary.json')
                memory = Path(visual_report['fused_memory'])
                current_scene = memory / 'scene'
            summary['rounds'][-1].update(status='passed', memory=str(memory),
                memory_version=read(memory / 'memory_state.json')['version'], visual_report=visual_report)
            write(root / 'pipeline_summary.json', summary)
        summary['base_graph_unchanged'] = digest([read(args.scene_dir / 'scene_graph.json'), read(args.scene_dir / 'scene_graph_edges.json')]) == base_hash
        summary['status'] = 'passed' if summary['base_graph_unchanged'] else 'failed'
    except Exception as exc:
        summary.update(status='failed', failed_stage=stage, error=str(exc))
        print(f'ERROR stage={stage}: {exc}', flush=True)
    write(root / 'pipeline_summary.json', summary)
    print(f"status={summary['status']} summary={root / 'pipeline_summary.json'}", flush=True)
    return 0 if summary['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
