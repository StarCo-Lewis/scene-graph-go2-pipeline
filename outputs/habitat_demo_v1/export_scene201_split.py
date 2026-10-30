"""Blender background script: split Scene201's kettle from its static stage.

Run only against the original house.blend. Never saves or changes that file.
"""
import bpy
import json
import sys
from mathutils import Vector
from pathlib import Path


def bounds_world(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    lo = [min(p[i] for p in corners) for i in range(3)]
    hi = [max(p[i] for p in corners) for i in range(3)]
    return [(a + b) / 2 for a, b in zip(lo, hi)], [b - a for a, b in zip(lo, hi)]


def export(path, selected_only):
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB",
                              use_selection=selected_only, export_yup=True,
                              export_apply=False)
    if not path.is_file() or path.stat().st_size < 1000:
        raise RuntimeError(f"GLB export failed: {path}")


def main():
    if "--" not in sys.argv:
        raise ValueError("Expected -- OUTPUT_DIRECTORY")
    extra = sys.argv[sys.argv.index("--") + 1:]
    out = Path(extra[0]).resolve()
    split_sideboard = "--sideboard" in extra[1:]
    out.mkdir(parents=True, exist_ok=True)
    kettle = bpy.data.objects.get("geometry_0.253")
    if kettle is None or kettle.type != "MESH":
        raise ValueError("Expected kettle mesh geometry_0.253 not found")
    centre, size = bounds_world(kettle)
    expected = Vector((-2.66718, 8.02967, 1.397))
    if (Vector(centre) - expected).length > 0.08 or not (0.1 < max(size) < 0.3):
        raise ValueError(f"Refusing to split unexpected geometry: centre={centre}, size={size}")
    original_parent = kettle.parent
    world_matrix = kettle.matrix_world.copy()
    kettle.parent = None
    kettle.matrix_world = world_matrix
    # Localize the render mesh around the known free-joint origin. The object
    # spawn position in Habitat is assigned separately in the execution demo.
    from mathutils import Matrix
    origin = Vector((-2.66718, 8.02967, 1.31381))
    kettle.matrix_world = Matrix.Translation(-origin) @ kettle.matrix_world
    bpy.ops.object.select_all(action="DESELECT")
    kettle.select_set(True)
    bpy.context.view_layer.objects.active = kettle
    export(out / "kettle_dynamic.glb", True)
    bpy.data.objects.remove(kettle, do_unlink=True)
    if original_parent and len(original_parent.children) == 0:
        bpy.data.objects.remove(original_parent, do_unlink=True)
    sideboard_names = []
    if split_sideboard:
        # These P_* meshes occupy the CG node-13 bbox and include the cabinet
        # body and doors. geometry_0.* meshes nearby are separate clutter.
        for obj in bpy.data.objects:
            if obj.type != "MESH" or not obj.name.startswith("P_"):
                continue
            c, size = bounds_world(obj)
            if (abs(c[0] + 5.77159) < 0.35 and abs(c[1] - 1.49913) < 0.75
                    and 0.0 < c[2] < 1.0):
                sideboard_names.append(obj.name)
        if len(sideboard_names) != 18 or "P_a4732598bed78f1f.001" not in sideboard_names:
            raise ValueError(f"Unexpected sideboard mesh set: {sideboard_names}")
        origin = Vector((-5.77159, 1.49913, 0.00711))
        for name in sideboard_names:
            obj = bpy.data.objects[name]
            world_matrix = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = Matrix.Translation(-origin) @ world_matrix
        bpy.ops.object.select_all(action="DESELECT")
        for name in sideboard_names:
            bpy.data.objects[name].select_set(True)
        bpy.context.view_layer.objects.active = bpy.data.objects[sideboard_names[0]]
        export(out / "sideboard_static.glb", True)
        for name in sideboard_names:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    bpy.ops.object.select_all(action="DESELECT")
    export(out / "scene201_without_kettle.glb", False)
    (out / "scene201_without_kettle.stage_config.json").write_text(json.dumps({
        "render_asset": "scene201_without_kettle.glb",
        "collision_asset": "scene201_without_kettle.glb",
        "up": [0, 1, 0], "front": [0, 0, -1], "origin": [0, 0, 0],
        "scale": [1, 1, 1], "units_to_meters": 1.0,
        "gravity": [0, -9.81, 0], "is_collidable": True,
        "requires_lighting": True, "shader_type": "material"
    }, indent=2), encoding="utf-8")
    (out / "kettle_dynamic.object_config.json").write_text(json.dumps({
        "render_asset": "kettle_dynamic.glb", "collision_asset": "kettle_dynamic.glb",
        "up": [0, 1, 0], "front": [0, 0, -1], "scale": [1, 1, 1],
        "units_to_meters": 1.0, "mass": 1.1, "margin": 0.005,
        "friction_coefficient": 0.6, "restitution_coefficient": 0.0,
        "use_bounding_box_for_collision": True, "join_collision_meshes": True,
        "semantic_id": 67
    }, indent=2), encoding="utf-8")
    if split_sideboard:
        (out / "sideboard_static.object_config.json").write_text(json.dumps({
            "render_asset": "sideboard_static.glb", "collision_asset": "sideboard_static.glb",
            "up": [0, 1, 0], "front": [0, 0, -1], "scale": [1, 1, 1],
            "units_to_meters": 1.0, "mass": 40.0,
            "friction_coefficient": 0.6, "restitution_coefficient": 0.0,
            "use_bounding_box_for_collision": False, "join_collision_meshes": True,
            "semantic_id": 13
        }, indent=2), encoding="utf-8")
        (out / "split_manifest.json").write_text(json.dumps({
            "source": bpy.data.filepath, "sideboard_meshes": sideboard_names,
            "nearby_geometry_meshes_not_split": ["geometry_0.092", "geometry_0.093", "geometry_0.094"],
            "note": "Original blend and scene GLB are unchanged"
        }, indent=2), encoding="utf-8")
    print("SCENE201_SPLIT=" + json.dumps({"kettle_centre_source": centre,
        "kettle_size_source": size, "stage_bytes": (out / "scene201_without_kettle.glb").stat().st_size,
        "kettle_bytes": (out / "kettle_dynamic.glb").stat().st_size,
        "sideboard_mesh_count": len(sideboard_names)}))


main()
