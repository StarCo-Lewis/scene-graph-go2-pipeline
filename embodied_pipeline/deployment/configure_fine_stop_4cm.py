"""Patch only the position fine-approach stop threshold; never starts ROS or motion.

Preview by default. --apply writes a timestamped backup then updates the file.
Unknown controller structures are rejected for review, never guessed.
"""
import argparse
import ast
from datetime import datetime
import difflib
from pathlib import Path


def patched_source(source):
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_position_cmd']
    if len(funcs) != 1:
        raise ValueError('Expected exactly one compute_position_cmd; inspect dog source first')
    tests = [n for n in ast.walk(funcs[0]) if isinstance(n, ast.If)
             and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
             and n.test.left.id == 'dist' and len(n.test.ops) == 1
             and isinstance(n.test.ops[0], ast.LtE)
             and any(isinstance(r, ast.Return) and isinstance(r.value, ast.Call)
                     and isinstance(r.value.func, ast.Name) and r.value.func.id == 'Cmd'
                     and len(r.value.args) == 5
                     and isinstance(r.value.args[3], ast.Constant) and r.value.args[3].value is True
                     and isinstance(r.value.args[4], ast.Constant) and r.value.args[4].value == 'reached'
                     for r in n.body)]
    if len(tests) != 1:
        raise ValueError('Unknown reached/stop branch; refusing modification')
    value = tests[0].test.comparators[0]
    if isinstance(value, ast.Constant) and type(value.value) is float and value.value == .04:
        return source
    allowed = [ast.dump(ast.parse(x, mode='eval').body) for x in ('cfg.pos_tol + cfg.coast','cfg.stop_radius_m','0.06','0.061')]
    if ast.dump(value) not in allowed:
        raise ValueError('Unrecognized old threshold: '+ast.get_source_segment(source, value))
    raw = source.encode('utf-8'); lines = raw.splitlines(keepends=True)
    a = sum(map(len, lines[:value.lineno-1])) + value.col_offset
    b = sum(map(len, lines[:value.end_lineno-1])) + value.end_col_offset
    updated = (raw[:a]+b'0.04'+raw[b:]).decode('utf-8')
    lines = updated.splitlines(keepends=True)
    # Remove adjacent old explanatory comments, which describe a different threshold.
    start = tests[0].lineno-1
    while start > funcs[0].lineno and lines[start-1].lstrip().startswith('#'):
        start -= 1
    lines[start:tests[0].lineno-1] = [
        '    # Operator update 2026-10-30: stop POSITION approach at <= 0.04 m.\n',
        '    # Fixed stage threshold; no coast added here. Speeds/yaw/final acceptance unchanged.\n',
        '    # Braking and subsequent turning can still change the final position.\n']
    updated = ''.join(lines)
    compile(updated, 'fine_approach.py', 'exec')
    return updated


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Dog nav/control/fine_approach.py')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    raw = args.source.read_bytes(); old = raw.decode('utf-8'); new = patched_source(old)
    if old == new:
        print('Already 0.04 m; no files changed. Running process not inspected.')
        return 0
    print(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
          fromfile=str(args.source), tofile=str(args.source))), end='')
    if args.apply:
        backup = args.source.with_name(args.source.name+'.before_4cm_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.bak')
        with backup.open('xb') as f:f.write(raw)
        if args.source.read_bytes() != raw:
            raise RuntimeError('Source changed during review; refusing overwrite')
        args.source.write_bytes(new.encode('utf-8'))
        print('\nBackup:',backup)
        print('Applied POSITION stop threshold 0.04 m. Restart navigation only when idle; no service restarted here.')
    else:
        print('\nPreview only. Use --apply to back up and update this exact source.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
