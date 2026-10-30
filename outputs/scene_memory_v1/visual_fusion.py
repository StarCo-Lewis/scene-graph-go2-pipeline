"""Known-instance, world-aligned RGB-D voxel overlay; never infer new objects."""
import copy
import math
from update_memory import digest


def project_pixel(u, v, depth, intrinsics, rotation, translation):
    if not math.isfinite(depth) or depth <= 0:
        raise ValueError('Depth must be finite positive axial metres')
    fx, fy, cx, cy = intrinsics
    if fx <= 0 or fy <= 0:
        raise ValueError('Invalid intrinsics')
    local = [(u + .5 - cx) * depth / fx, -(v + .5 - cy) * depth / fy, -depth]
    world = [sum(rotation[i][j] * local[j] for j in range(3)) + translation[i] for i in range(3)]
    return [world[0], -world[2], world[1]]  # Habitat -> ConceptGraphs


def fuse_frame(previous, state, frame_id, points_by_id, voxel_m=.03):
    if not math.isfinite(voxel_m) or voxel_m <= 0:
        raise ValueError('Invalid voxel size')
    layer = copy.deepcopy(previous) if previous else {'schema_version': '1.0',
        'coordinate_frame': 'conceptgraphs_world', 'voxel_m': voxel_m,
        'association': 'simulator_semantic_id_oracle_not_VLM', 'frames': {}, 'objects': {}}
    if layer['voxel_m'] != voxel_m:
        raise ValueError('Voxel size changed')
    fingerprint = digest({'points': points_by_id, 'state_version': state['version'],
                          'poses': {k: e['node'].get('bbox_center') for k, e in state['entities'].items()}})
    if frame_id in layer['frames']:
        if layer['frames'][frame_id] != fingerprint:
            raise ValueError('Conflicting visual frame ID')
        return layer
    # Tombstones or movement invalidate stale object points, even when unseen.
    for key in list(layer['objects']):
        entity = state['entities'].get(key)
        if not entity or not entity['active'] or layer['objects'][key]['pose_digest'] != digest(entity['node']['bbox_center']):
            del layer['objects'][key]
    for key, points in points_by_id.items():
        entity = state['entities'].get(str(key))
        if not entity or not entity['active']:
            raise ValueError('NO object can be used in this task')
        if not points:
            continue
        centre = entity['node']['bbox_center']
        extent = entity['node'].get('bbox_extent', [.5, .5, .5])
        limit = math.sqrt(sum(float(x)**2 for x in extent)) / 2 + .25
        accepted = []
        for point in points:
            if len(point) != 6 or not all(math.isfinite(float(x)) for x in point):
                raise ValueError('Invalid XYZRGB point')
            if math.dist(point[:3], centre) <= limit:
                accepted.append(point)
        if len(accepted) < 20:
            continue  # insufficient or conflicting geometry: preserve prior layer
        entry = layer['objects'].setdefault(str(key), {'pose_digest': digest(centre), 'voxels': {}, 'frame_ids': []})
        for point in accepted:
            voxel = ','.join(str(math.floor(x / voxel_m)) for x in point[:3])
            old = entry['voxels'].get(voxel)
            if old:
                n = old['count']
                old['mean_xyzrgb'] = [(old['mean_xyzrgb'][i]*n + point[i])/(n+1) for i in range(6)]
                old['count'] += 1
            else:
                entry['voxels'][voxel] = {'mean_xyzrgb': point, 'count': 1}
        entry['frame_ids'].append(frame_id)
    layer['frames'][frame_id] = fingerprint
    return layer
