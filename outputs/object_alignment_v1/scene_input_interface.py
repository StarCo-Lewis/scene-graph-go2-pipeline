"""Validate upstream scene input without changing downstream plan formats."""
import copy
import json
import math


def normalize_scene_input(scene_graph_json, instruction):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('instruction must be a nonempty string')
    data = json.loads(scene_graph_json) if isinstance(scene_graph_json, str) else scene_graph_json
    # A dict wrapper is supported explicitly; no recursive/guessed extraction.
    if isinstance(data, dict):
        data = data.get('nodes')
    if not isinstance(data, list):
        raise ValueError('scene_graph_json must be a node array or a nodes wrapper')
    nodes = copy.deepcopy(data)
    seen = set()
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError('Each scene node must be an object')
        node_id = node.get('id')
        if isinstance(node_id, bool) or not isinstance(node_id, int) or node_id in seen:
            raise ValueError('Node id must be a unique integer')
        seen.add(node_id)
        for name in ('object_tag', 'caption'):
            if not isinstance(node.get(name), str):
                raise ValueError(name + ' must be a string')
        if not node['object_tag'].strip():
            raise ValueError('object_tag must not be empty')
        for name in ('bbox_center', 'bbox_extent'):
            values = node.get(name)
            if not isinstance(values, list) or len(values) != 3 or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values
            ):
                raise ValueError(name + ' must contain three finite numbers')
        if any(x <= 0 for x in node['bbox_extent']):
            raise ValueError('bbox_extent must contain positive full dimensions')
    return {'instruction': instruction, 'nodes': nodes}
