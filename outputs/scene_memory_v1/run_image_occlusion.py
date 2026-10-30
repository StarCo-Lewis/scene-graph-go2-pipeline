"""Four fixed-checkpoint input-dropout cases; no graph mutation or physical occluder."""
import subprocess
from pathlib import Path
from datetime import datetime
from run_full_multiround import HERE, SCENE
from update_memory import read, write, digest


def assess_case(name, result, code):
    accepted = sum(f.get('status') == 'accepted' for f in result.get('frames', []))
    if result.get('status') not in ('non_oracle_visual_evidence_ready', 'insufficient_visual_evidence') or code not in (0, 2):
        return {'passed': False, 'reason': 'API/runtime failure, not evidence rejection', 'accepted_views': accepted}
    if name == 'fully_unobservable':
        passed = accepted == 0 and result['status'] == 'insufficient_visual_evidence'
    elif name.startswith('partial_dropout'):
        passed = result['status'] in ('non_oracle_visual_evidence_ready', 'insufficient_visual_evidence')
    else:
        passed = result['status'] == 'non_oracle_visual_evidence_ready' and accepted >= 2
    return {'passed': passed and result.get('source_memory_unchanged') is True,
            'accepted_views': accepted, 'evidence_status': result['status'],
            'partial_visibility_success': name.startswith('partial_dropout') and result['status'] == 'non_oracle_visual_evidence_ready'}


def run(args):
    source = args.source_run
    origin = read(source / 'regression_summary.json')
    if origin.get('status') != 'passed':
        raise ValueError('A passed long-workflow source required')
    memory = Path(origin['latest_pre_perturbation_memory'])
    before = digest(read(memory / 'memory_state.json'))
    root = args.output_dir or SCENE / 'image_occlusion_validation' / datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'running', 'training_eligible': False, 'cases': [], 'source_memory': str(memory),
        'scope': 'Fixed-checkpoint RGB-D dropout and visual evidence recovery; no pose update or physical occlusion',
        'limitations': ['Partial test region uses prior SAM mask only to construct fixture',
                        'Full dropout blanks entire image and invalidates depth, not merely the kettle',
                        'Memory immutability is tested in perception-only replay, not action commit/delete integration',
                        'Recovery reuses original visible inputs; no autonomous viewpoint selection']}
    modalities = ('rgb', 'depth', 'joint') if getattr(args, 'factorial_dropout', False) else ('joint',)
    conditions = [('visible_control', 'none', None, 'joint')]
    conditions += [(f'partial_dropout_{modality}_{int(fraction*100)}', 'partial', fraction, modality)
                   for modality in modalities for fraction in (.25, .5, .75)]
    conditions += [('fully_unobservable', 'full', None, 'joint'), ('visibility_restored', 'none', None, 'joint')]
    summary['factorial_dropout'] = bool(getattr(args, 'factorial_dropout', False))
    summary['limitations'].append('RGB-only fill retains original depth: modality ablation, not occluder geometry; depth-only preserves RGB')
    for name, mode, fraction, modality in conditions:
        command = [args.planning_python, str(HERE / 'replay_vlm_rgbd.py'),
            '--visual-dir', str(source / 'baseline_capture'), '--memory-dir', str(memory),
            '--max-views', '3', '--test-occlusion', mode,
            '--occlusion-fixture', str(source / 'baseline_detection'), '--output-dir', str(root / name)]
        if fraction is not None:
            command += ['--occlusion-fraction', str(fraction)]
        command += ['--dropout-modality', modality]
        print(f'START case={name}', flush=True)
        with (root / (name+'.console.log')).open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            code = process.wait()
        report = root / name / 'regression_summary.json'
        result = read(report) if report.exists() else {'status': 'missing_report'}
        row = {'case': name, 'dropout_modality': modality, 'target_bbox_width_fraction': fraction,
               'exit_code': code, **assess_case(name, result, code)}
        row['frame_results'] = result.get('frames', [])
        row['category_recognized_view_count'] = result.get('category_recognized_view_count')
        row['category_only_view_count'] = result.get('category_only_view_count')
        row['memory_unchanged_after_case'] = digest(read(memory / 'memory_state.json')) == before
        summary['cases'].append(row)
        write(root / 'regression_summary.json', summary)
    summary['status'] = 'passed' if all(c['passed'] and c['memory_unchanged_after_case'] for c in summary['cases']) else 'needs_review'
    summary['active_kettle_preserved'] = read(memory / 'memory_state.json')['entities']['67']['active'] is True
    summary['partial_observation_success_count'] = sum(c.get('partial_visibility_success', False) for c in summary['cases'])
    summary['interpretation'] = 'Passed means safe handling and restored control, not all partial dropout levels recognized'
    if not summary['active_kettle_preserved']:
        summary['status'] = 'failed'
    write(root / 'regression_summary.json', summary)
    print(f"status={summary['status']} summary={root / 'regression_summary.json'}", flush=True)
    return 0 if summary['status'] == 'passed' else 2
