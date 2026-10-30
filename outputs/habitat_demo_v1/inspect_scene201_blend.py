"""Read-only Blender inventory for the Scene201 kettle and target furniture."""
import bpy
import json
from mathutils import Vector

terms = ("kettle", "credenza", "sideboard")
matches = []
for obj in bpy.data.objects:
    if any(term in obj.name.lower() for term in terms):
        matches.append({"name": obj.name, "type": obj.type,
                        "parent": obj.parent.name if obj.parent else None,
                        "location": [round(float(v), 4) for v in obj.location],
                        "children": len(obj.children),
                        "mesh_vertices": len(obj.data.vertices) if obj.type == "MESH" else None})
print("SCENE201_OBJECT_COUNT=" + str(len(bpy.data.objects)))
print("SCENE201_SAMPLE_NAMES=" + json.dumps([obj.name for obj in list(bpy.data.objects)[:25]], ensure_ascii=True))
print("SCENE201_ASSET_INVENTORY=" + json.dumps(matches, ensure_ascii=True))
for label, target in {"kettle": (-2.66718, 8.02967, 1.31381),
                      "sideboard": (-5.77159, 1.49913, 0.00711)}.items():
    near = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        centre = sum((obj.matrix_world @ Vector(corner) for corner in obj.bound_box), Vector()) / 8
        distance = (centre - Vector(target)).length
        near.append((distance, obj.name, tuple(round(float(x), 3) for x in centre),
                     tuple(round(float(x), 3) for x in obj.dimensions),
                     obj.parent.name if obj.parent else None))
    near.sort()
    print("SCENE201_NEAR_" + label.upper() + "=" + json.dumps(near[:15], ensure_ascii=True))
for name in ("geometry_0.253", "geometry_0.254", "world.265", "world.266"):
    obj = bpy.data.objects.get(name)
    if obj:
        print("SCENE201_DETAIL=" + json.dumps({
            "name": name, "type": obj.type, "parent": obj.parent.name if obj.parent else None,
            "children": [x.name for x in obj.children],
            "collections": [x.name for x in obj.users_collection],
            "custom_properties": {key: str(obj[key])[:200] for key in obj.keys()},
            "materials": [m.name if m else None for m in obj.data.materials] if obj.type == "MESH" else [],
            "images": [n.image.name + ":" + n.image.filepath for m in obj.data.materials if m and m.use_nodes
                       for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image]
                      if obj.type == "MESH" else [],
        }, ensure_ascii=True))
for name in ("Image_401", "Image_402"):
    image = bpy.data.images.get(name)
    if image:
        print("SCENE201_IMAGE=" + json.dumps({"name": name, "size": list(image.size),
            "source": image.source, "packed": bool(image.packed_file)}, ensure_ascii=True))
