"""Ten realistic, location-conditioned long dialogues; proxy feedback, not physics."""
import argparse
import logging
from datetime import datetime
from pathlib import Path

import run_dialogue_chain as dialogue
import run_crossroom_task as pilot
from realistic_long_cases import CASE_VERSION,episodes,evaluate
from long_task_error_audit import build as build_error_audit


def run(args):
    args.memory_mode='execution_feedback'
    args.episode_provider=episodes
    args.feedback_evaluator=evaluate
    args.case_version=CASE_VERSION
    args.suite='realistic_long10_varied_user_locations'
    args.reworded_rounds=[]
    args.comparison_note=('Ten different task families, five per scene, with declared user locations. '
        'Not matched tasks or a model/scene/location causal comparison. Independent initial states.')
    code=dialogue.run(args)
    build_error_audit(args.output_dir.resolve())
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--episode-ids',nargs='+',choices=[e['episode_id'] for e in episodes()])
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--inference-mode',choices=('baseline','bounded_inference'),default='bounded_inference')
    parser.add_argument('--alignment-wire-format',choices=('matrix','legacy'),default='matrix',
        help='Matrix decomposition and object binding by default; skill actions stay four-column matrices')
    parser.add_argument('--matrix-contract',choices=('typed5_schema','typed9_schema','strict_schema','typed_json','legacy_json'))
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--model',default='qwen3.8-flash')
    parser.add_argument('--base-url',default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    parser.add_argument('--embedding-python',type=Path,default=Path('/home/lewis/miniforge3/envs/skill_embed/bin/python'))
    parser.add_argument('--embedding-model-dir',type=Path,default=Path('/mnt/d/NebulaAI/models/skill_embedding_minilm_v1'))
    args=parser.parse_args()
    if args.matrix_contract is None:
        args.matrix_contract='typed9_schema' if args.alignment_wire_format=='matrix' else 'legacy_json'
    if args.output_dir is None:
        prefix = ('long10_v7_typed5_' if args.matrix_contract=='typed5_schema' else
                  'long10_v6_typed9_' if args.matrix_contract=='typed9_schema' else
                  'long10_v5_typed_matrix_' if args.matrix_contract!='legacy_json' else
                  'long10_matrix_' if args.alignment_wire_format == 'matrix' else 'long10_')
        args.output_dir = pilot.HERE/'runs'/datetime.now().strftime(prefix+'%Y%m%d_%H%M%S_%f')
    args.top_k,args.cosine_threshold,args.cosine_margin=2,.35,.05
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
