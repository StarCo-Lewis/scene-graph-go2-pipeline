"""Explicit position conventions; no map registration or robot-state inference."""

import math

from .real_rgbd import WORLD

PROXY_WORLD = "proxy_world_y_up_same_origin"


def vector3(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError("Expected three finite metric coordinates")
    return [float(v) for v in value]


def position(value, source, target):
    x, y, z = vector3(value)
    if source == WORLD and target == PROXY_WORLD:
        return [x, z, -y]
    if source == PROXY_WORLD and target == WORLD:
        return [x, -z, y]
    if source == target and source in (WORLD, PROXY_WORLD):
        return [x, y, z]
    raise ValueError("Unknown world frame; no implicit coordinate conversion")


def extent(value, source, target):
    dims = vector3(value)
    if min(dims) <= 0:
        raise ValueError("Bounding-box full dimensions must be positive")
    if source == target and source in (WORLD, PROXY_WORLD):
        return dims
    if {source, target} == {WORLD, PROXY_WORLD}:
        return [dims[0], dims[2], dims[1]]
    raise ValueError("Unknown extent frame")


def contract():
    return dict(schema_version="real_scene_axis_adapter_v1", source_frame=WORLD,
        target_frame=PROXY_WORLD, unit="metre", shared_origin=True,
        position_rule="[x,y,z] -> [x,z,-y]", inverse_rule="[x,y,z] -> [x,-z,y]",
        full_extent_rule="[dx,dy,dz] -> [dx,dz,dy]",
        world_transform=[[1,0,0,0], [0,0,1,0], [0,-1,0,0], [0,0,0,1]],
        pose_rule="T_target_camera = world_transform @ T_source_camera; camera basis unchanged",
        execution_library_rule="Raw graph boxes stay Z-up; only runtime positions use the adapter, as in existing proxy initialization",
        physical_habitat_registration_established=False, robot_to_map_localization_established=False,
        applied_to_graph=False, runtime_state_created=False)
