"""Process-isolated prompt selection around the unchanged alignment engine."""

import argparse
import sys
from contextlib import nullcontext

from .._legacy import load
from .intent_prompt import ARMS, compose
from .process_contract import compose_process_prompt, installed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--prompt-arm", choices=ARMS, required=True)
    parser.add_argument("--process-constraints", action="store_true")
    args, rest = parser.parse_known_args(argv)
    engine = load("align_objects")
    original_prompt, original_argv = engine.DECOMPOSITION_PROMPT, sys.argv
    try:
        engine.DECOMPOSITION_PROMPT = compose(original_prompt, args.prompt_arm)
        if args.process_constraints:
            engine.DECOMPOSITION_PROMPT = compose_process_prompt(engine.DECOMPOSITION_PROMPT)
        sys.argv = ["intent_alignment_worker"] + rest
        with installed() if args.process_constraints else nullcontext():
            return engine.main()
    finally:
        engine.DECOMPOSITION_PROMPT = original_prompt
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
