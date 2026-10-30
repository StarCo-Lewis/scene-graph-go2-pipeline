"""Inspect safely, or explicitly launch an existing suite through modular flow."""

import json
import sys

from .introspection import describe, verify_legacy


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "field-console":
        from .ui.field_console import main as console_main
        return console_main(argv[1:])
    if argv and argv[0] == "dispatch-preview":
        from .execution.controller_dispatch import main as dispatch_main
        return dispatch_main(argv[1:])
    if not argv or argv == ["inspect"]:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
        return 0
    if argv == ["verify-legacy"]:
        result = verify_legacy()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "unchanged" else 2
    if argv[0] in ("-h", "--help"):
        print("Usage: python -m embodied_pipeline {inspect|verify-legacy|dialogue|matrix-followup|oracle-decomposition|intent-comparison|real-scene|real-scene-plan|navigation-replay|field-console|dispatch-preview} [runner options]")
        print("No arguments is read-only. --preflight-only prepares inputs without model calls.")
        print("dialogue/matrix-followup without --preflight-only are explicit LIVE launches.")
        print("Runner defaults are unchanged; select --matrix-contract typed5_schema explicitly.")
        return 0
    if argv[0] == "oracle-decomposition":
        from .orchestration.oracle_decomposition import main as oracle_main
        return oracle_main(argv[1:])
    if argv[0] == "intent-comparison":
        from .orchestration.intent_comparison import main as intent_main
        return intent_main(argv[1:])
    if argv[0] == "real-scene-plan":
        from .orchestration.real_scene_plan import main as plan_main
        return plan_main(argv[1:])
    if argv[0] == "navigation-replay":
        from .orchestration.navigation_replay import main as replay_main
        return replay_main(argv[1:])
    if argv[0] == "real-scene":
        from .orchestration.real_scene import main as real_main
        return real_main(argv[1:])
    if argv[0] not in ("dialogue", "matrix-followup"):
        print("Unknown command. Use --help.", file=sys.stderr)
        return 2
    command = argv.pop(0)
    from .orchestration import batch, matrix_followup
    workflow = batch if command == "dialogue" else matrix_followup
    previous = sys.argv
    try:
        sys.argv = [f"embodied_pipeline {command}"] + argv
        return workflow.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())

