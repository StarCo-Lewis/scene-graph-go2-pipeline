"""Offline source patch for the navigation group's Action server; never starts ROS.

Default: display the exact diff. --apply: backup and update the specified file.
First goal still establishes T; later goals cannot recreate T on start mismatch.
"""
import argparse
import ast
from datetime import datetime
import difflib
from pathlib import Path


def patched_source(source):
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RosActionServer"]
    if len(classes) != 1:
        raise ValueError("Expected one RosActionServer class; refusing an unknown source layout")
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == "_execute"]
    if len(methods) != 1:
        raise ValueError("Expected one _execute method")
    calls = [n for n in ast.walk(methods[0]) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "ensure_aligned"
             and ast.dump(n.func.value) == ast.dump(ast.parse("self.nav", mode="eval").body)]
    if len(calls) != 1:
        raise ValueError("Expected one self.nav.ensure_aligned call; refusing an unknown implementation")
    values = [k.value for k in calls[0].keywords if k.arg == "realign_on_mismatch"]
    if len(values) != 1:
        raise ValueError("Missing explicit realign_on_mismatch argument")
    value = values[0]
    if isinstance(value, ast.Constant) and value.value is False:
        return source
    if ast.dump(value) != ast.dump(ast.parse("self.realign_on_mismatch", mode="eval").body):
        raise ValueError("Unexpected realignment policy; review manually")
    # AST column offsets are UTF-8 byte offsets, including in Chinese source files.
    lines = source.encode("utf-8").splitlines(keepends=True)
    begin = sum(map(len, lines[:value.lineno-1])) + value.col_offset
    end = sum(map(len, lines[:value.end_lineno-1])) + value.end_col_offset
    data = source.encode("utf-8")
    updated = (data[:begin] + b"False" + data[end:]).decode("utf-8")
    compile(updated, "ros_action.py", "exec")
    return updated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ros_action", type=Path, help="Path to navigation repository mission/ros_action.py")
    parser.add_argument("--apply", action="store_true", help="Write a backup and patch source; does not restart any service")
    args = parser.parse_args(argv)
    raw = args.ros_action.read_bytes()
    source = raw.decode("utf-8")
    updated = patched_source(source)
    if updated == source:
        print("Already first-alignment-only. No files changed; running service not inspected.")
        return 0
    print("".join(difflib.unified_diff(source.splitlines(True), updated.splitlines(True),
                                    fromfile=str(args.ros_action), tofile=str(args.ros_action))), end="")
    if args.apply:
        backup = args.ros_action.with_name(args.ros_action.name + ".before_first_alignment_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".bak")
        with backup.open("xb") as stream:
            stream.write(raw)
        # Never overwrite a concurrent edit after reviewing its original content.
        if args.ros_action.read_bytes() != raw:
            raise RuntimeError("Source changed during preparation; not overwritten")
        args.ros_action.write_bytes(updated.encode("utf-8"))
        print("Backup:", backup)
        print("Source updated only. Arrange a stopped, idle service restart before the NEXT navigation session.")
    else:
        print("Preview only. Use --apply to update source after reviewing this diff.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
