#!/usr/bin/env python3
"""Translation-only trimmed nearest-point registration; evaluation truth is not an input."""
import argparse
import math
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path
from update_memory import read, write, digest


def sample(points, maximum=160):
    if len(points) < 32 or any(len(p) < 3 or any(not math.isfinite(float(x)) for x in p[:3]) for p in points):
        raise ValueError('Insufficient finite point cloud')
    step = max(1, math.ceil(len(points)/maximum))
    return [[float(x) for x in p[:3]] for p in points[::step]]


def register_translation(before, after):
    source, target = sample(before), sample(after)
    delta = [statistics.median(p[i] for p in target)-statistics.median(p[i] for p in source) for i in range(3)]
    for _ in range(30):
        pairs = []
        for p in source:
            shifted = [p[i]+delta[i] for i in range(3)]
            q = min(target, key=lambda row: math.dist(shifted, row))
            pairs.append((math.dist(shifted, q), p, q))
        pairs.sort(key=lambda row: row[0])
        retained = pairs[:max(16, int(len(pairs)*.8))]
        candidate = [sum(q[i]-p[i] for _, p, q in retained)/len(retained) for i in range(3)]
        change = math.dist(candidate, delta)
        delta = candidate
        if change < 1e-5:
            break
    shifted = [[p[i]+delta[i] for i in range(3)] for p in source]
    forward = [min(math.dist(p, q) for q in target) for p in shifted]
    reverse = [min(math.dist(p, q) for p in shifted) for q in target]
    residuals = sorted(forward + reverse)
    retained = residuals[:int(len(residuals)*.8)]
    return {'translation_cg': delta, 'symmetric_trimmed_rms_m': math.sqrt(sum(d*d for d in retained)/len(retained)),
            'method': 'translation_only_trimmed_nearest_point_ICP', 'rotation_estimated': False}


def consensus(estimates):
    from itertools import combinations
    accepted = [e for e in estimates if e['symmetric_trimmed_rms_m'] <= .035]
    if len(accepted) < 2:
        raise ValueError('At least two views must pass residual <=0.035m')
    delta = [statistics.median(e['translation_cg'][i] for e in accepted) for i in range(3)]
    spread = max(math.dist(e['translation_cg'], delta) for e in accepted)
    chosen = tuple(range(len(accepted)))
    if spread > .05:
        if len(accepted) > 12:
            raise ValueError('Too many candidates for bounded consensus search')
        winners = []
        for size in range(len(accepted)-1, max(2, len(accepted)//2), -1):
            for indices in combinations(range(len(accepted)), size):
                centre = [statistics.median(accepted[j]['translation_cg'][i] for j in indices) for i in range(3)]
                if max(math.dist(accepted[j]['translation_cg'], centre) for j in indices) <= .05:
                    winners.append((indices, centre))
            if winners:
                break
        if len(winners) != 1:
            raise ValueError('Inconsistent view translations: no unique strict-majority group of at least three views')
        chosen, delta = winners[0]
        spread = max(math.dist(accepted[j]['translation_cg'], delta) for j in chosen)
    if not .05 <= math.sqrt(sum(v*v for v in delta)) <= .75:
        raise ValueError('Displacement outside 0.05..0.75m')
    return {'translation_cg': delta, 'max_view_spread_m': spread, 'accepted_views': len(chosen),
            'residual_qualified_views': len(accepted),
            'consensus_policy': 'all_consistent_or_unique_strict_majority_minimum_three',
            'selected_frames': [accepted[j].get('frame', str(j)) for j in chosen],
            'excluded_frames': [e.get('frame', str(j)) for j, e in enumerate(accepted) if j not in chosen]}


def baseline_points(visual_dir, detections_dir, name):
    import numpy as np
    from PIL import Image
    from visual_fusion import project_pixel
    metadata = read(visual_dir / (name+'.json'))
    mask = np.asarray(Image.open(detections_dir / (name+'.mask.png'))) > 0
    with np.load(visual_dir / (name+'.npz'), allow_pickle=False) as archive:
        depth = archive['depth'].copy()  # no semantic field accessed
    valid = mask & np.isfinite(depth) & (depth > .05) & (depth < 10)
    if valid.sum() < 32:
        raise ValueError('Insufficient baseline depth')
    median = float(np.median(depth[valid]))
    mad = float(np.median(np.abs(depth[valid]-median)))
    valid &= np.abs(depth-median) <= max(.08, 3*mad)
    ys, xs = np.where(valid)
    return [project_pixel(int(x), int(y), float(depth[y,x]), metadata['intrinsics_fx_fy_cx_cy'],
        metadata['camera_rotation_matrix'], metadata['camera_position_habitat']) for y, x in zip(ys, xs)]


def estimate(args):
    reports = []
    for point_path in sorted(args.after_detections.glob('view_*.points.json')):
        name = point_path.name.split('.')[0]
        before_meta = read(args.before_visual / (name+'.json'))
        after_meta = read(args.after_visual / (name+'.json'))
        # Match fixed-view observations. This pilot does not estimate camera motion.
        for key in ('camera_rotation_matrix', 'camera_position_habitat', 'intrinsics_fx_fy_cx_cy'):
            if digest(before_meta[key]) != digest(after_meta[key]):
                raise ValueError('Camera calibration/pose differs between paired views')
        if not (args.before_detections / (name+'.mask.png')).exists():
            continue
        before = baseline_points(args.before_visual, args.before_detections, name)
        after = read(point_path)['points_xyzrgb']
        reports.append(dict(register_translation(before, after), frame=name))
    result = dict(consensus(reports), status='translation_estimated', views=reports,
        oracle_translation_read=False, semantic_labels_used=False, memory_bbox_centre_used_for_registration=False,
        scope='Fixed camera, single known kettle, translation only; no rotation or general instance re-identification')
    write(args.output, result)
    print(f'status=translation_estimated result={args.output}', flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('before_visual', 'before_detections', 'after_visual', 'after_detections', 'output'):
        p.add_argument('--'+key.replace('_', '-'), type=Path, required=True)
    args = p.parse_args()
    try:
        estimate(args)
    except Exception as exc:
        write(args.output, {'status': 'rejected', 'error': str(exc), 'pose_update_allowed': False})
        raise
