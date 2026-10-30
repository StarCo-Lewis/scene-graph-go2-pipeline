"""Scene202 two tasks + Scene200 six tasks; flat matrix symbolic tests only."""
import argparse
import logging
from datetime import datetime
from pathlib import Path

import run_crossroom_task as pilot
from run_flat_stress import run_cases
from two_scene_stress_cases import scene202_cases,scene200_cases


def inputs():
    sources=[pilot.HERE/'scenarios/scene202_crossroom_6.json',
             pilot.HERE/'scenarios/scene200_symbolic_template.json']
    bases=[pilot.read_json(p) for p in sources]
    for p,base in zip(sources,bases):
        base['source_scene_dir']=str((p.parent/base['source_scene_dir']).resolve())
    return scene202_cases(bases[0])+scene200_cases(bases[1]),sources


def main():
    entries,sources=inputs()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',action='append',choices=[name for name,_ in entries])
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=pilot.HERE/'runs'/datetime.now().strftime('two_scene_%Y%m%d_%H%M%S_%f'))
    parser.add_argument('--model',default='qwen3.8-flash')
    parser.add_argument('--base-url',default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    parser.add_argument('--embedding-python',type=Path,default=Path('/home/lewis/miniforge3/envs/skill_embed/bin/python'))
    parser.add_argument('--embedding-model-dir',type=Path,default=Path('/mnt/d/NebulaAI/models/skill_embedding_minilm_v1'))
    args=parser.parse_args()
    if not 1<=args.repeats<=3:
        parser.error('repeats must be 1..3; first boundary test should use one')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
    return run_cases(args,entries,sources,
        'Eight authored positive tasks on two VLM-generated full-scene fixtures: Scene202 two, Scene200 six. Unequal difficulty/sample allocation; descriptive robustness test, not cross-scene accuracy or causal comparison.')


if __name__=='__main__': raise SystemExit(main())
