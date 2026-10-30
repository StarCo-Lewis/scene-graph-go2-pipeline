"""Read-only Blender inventory for the CG sideboard node 13."""
import bpy
import json
from mathutils import Vector

rows = []
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    pts = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    centre = [(a+b)/2 for a,b in zip(lo, hi)]
    size = [b-a for a,b in zip(lo, hi)]
    if abs(centre[0] + 5.77159) < 1.0 and abs(centre[1] - 1.49913) < 1.4 and 0 < centre[2] < 1.3:
        chain = []
        parent = obj.parent
        while parent is not None and len(chain) < 5:
            chain.append(parent.name)
            parent = parent.parent
        rows.append({"name": obj.name, "centre": [round(float(x),3) for x in centre],
                     "size": [round(float(x),3) for x in size], "parents": chain})
print("SIDEBOARD_REGION=" + json.dumps(sorted(rows, key=lambda r: r["centre"][2])))
