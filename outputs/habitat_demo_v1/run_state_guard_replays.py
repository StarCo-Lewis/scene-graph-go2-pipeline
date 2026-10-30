"""Two saved failing ORIGINAL rounds, independently replanned; not an 18-round pass."""
import argparse
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from run_navigation_demo import read_json, write_json
from run_json_long_horizon import SCENE
from replay_json_round_planning import INPUT_FILES, validate_checkpoint, checkpoint_hashes
from run_live_llm_sequence import execute_worker
from run_json_stability_batch import freeze_engine, frozen_unchanged, DEFAULT_URL
from strict_prompt_v2 import hierarchical_v2
from token_usage_audit import usage

SOURCE = SCENE/'hierarchical_json_ab/scene201_20261004_145912_513615'
CASES = {
    'holding_constraint': 'extended_mixed_18_repeat_1_v2/round_16/planning',
    'duplicate_approach': 'extended_mixed_18_repeat_2_v2/round_5/planning',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-batch', type=Path, default=SOURCE)
    p.add_argument('--output-dir', type=Path, default=SCENE/'state_guard_replays'/datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f'))
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--compare-v2', action='store_true', help='Also fresh-replan unguarded v2; doubles cases')
    p.add_argument('--model', default='qwen3.8-flash')
    p.add_argument('--base-url', default=DEFAULT_URL)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--unguarded', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    if not args.preflight_only and not os.environ.get('DASHSCOPE_API_KEY', '').strip():
        p.error('DASHSCOPE_API_KEY required')
    if args.worker:
        return hierarchical_v2(args, args.output_dir, {'max_depth':3, 'max_nodes':23, 'max_leaves':12},
                               state_guards=not args.unguarded)
    sources = {}
    for name, rel in CASES.items():
        source = args.source_batch/'trials'/rel
        state, descriptor, library, nodes, hashes = validate_checkpoint(source)
        sources[name] = (source, hashes, descriptor['round'], state['backend']['memory_version'])
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    manifest = freeze_engine(root/'frozen_engine')
    summary = {'status':'running', 'mode':'independent_saved_round_replay', 'trials':[],
               'simulation_executed':False, 'geometry_verified':False, 'training_eligible':False,
               'completed_chain_claim':False, 'historical_results_unchanged':True,
               'scope':'State-boundary implementation experiment, not prompt-only A/B. Two old failing round inputs.'}
    try:
        for index, (name, (source, hashes, number, version)) in enumerate(sources.items()):
            profiles = ['v2', 'v2_state_guards'] if args.compare_v2 else ['v2_state_guards']
            if index % 2:
                profiles.reverse()
            for profile in profiles:
                folder = root/f'{name}_{profile}'
                for filename in INPUT_FILES:
                    dest = folder/filename
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source/filename, dest)
                if checkpoint_hashes(folder) != hashes:
                    raise ValueError('Copied checkpoint mismatch')
                row = {'case':name, 'profile':profile, 'source':str(source), 'round':number,
                       'memory_version':version, 'input_hashes':hashes, 'output_dir':str(folder)}
                summary['trials'].append(row)
                write_json(root/'replay_summary.json', summary)
                if args.preflight_only:
                    row['status'] = 'input_preflight_passed'
                    continue
                started = time.monotonic()
                print(f'START {name} {profile} round={number}', flush=True)
                try:
                    if not frozen_unchanged(root/'frozen_engine', manifest):
                        raise ValueError('Frozen engine changed')
                    cmd = [sys.executable, str(root/'frozen_engine/outputs/habitat_demo_v1/run_state_guard_replays.py'),
                           '--worker', '--output-dir', str(folder), '--model', args.model, '--base-url', args.base_url]
                    if profile == 'v2':
                        cmd.append('--unguarded')
                    execute_worker(cmd, folder/'console.log', timeout=1890)
                    result = read_json(folder/'planning_summary.json')
                    if result['status'] != 'planning_passed' or result['input_memory_version'] != version:
                        raise ValueError('Replay status/version mismatch')
                    row['status'] = 'single_round_symbolic_passed'
                except Exception as exc:
                    row.update(status='failed', error=str(exc))
                    if (folder/'planning_summary.json').exists():
                        row['planning_error'] = read_json(folder/'planning_summary.json').get('error')
                row.update(seconds=time.monotonic()-started, token_usage=usage(folder))
                if checkpoint_hashes(source) != hashes:
                    raise ValueError('Historical checkpoint changed')
                print(f'END {name} {profile} status={row["status"]}', flush=True)
                write_json(root/'replay_summary.json', summary)
        summary['status'] = 'input_preflight_passed' if args.preflight_only else 'completed'
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status='interrupted_or_invalid', error=str(exc))
    finally:
        summary['frozen_engine_unchanged'] = frozen_unchanged(root/'frozen_engine', manifest)
        summary['historical_results_unchanged'] = all(checkpoint_hashes(s) == h for s,h,_,_ in sources.values())
        if not summary['frozen_engine_unchanged'] or not summary['historical_results_unchanged']:
            summary['status'] = 'invalid'
        write_json(root/'replay_summary.json', summary)
    print(f'status={summary["status"]} summary={root/"replay_summary.json"}', flush=True)
    return 0 if summary['status'] == 'input_preflight_passed' or (summary['status'] == 'completed' and all(
        r.get('status') == 'single_round_symbolic_passed' for r in summary['trials'])) else 2


if __name__ == '__main__':
    raise SystemExit(main())
