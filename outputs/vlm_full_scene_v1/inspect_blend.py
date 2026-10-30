"""Read-only inventory. Run inside Blender with -- OUTPUT.json."""
import json
import sys
from pathlib import Path
import bpy

out = Path(sys.argv[sys.argv.index('--') + 1])
objects = list(bpy.context.scene.objects)
rows = [{'name': o.name, 'type': o.type, 'parent': o.parent.name if o.parent else None,
         'children': len(o.children), 'location': list(o.matrix_world.translation),
         'dimensions': list(o.dimensions)} for o in objects]
report = {'source': bpy.data.filepath, 'object_count': len(rows),
          'mesh_count': sum(o.type == 'MESH' for o in objects),
          'root_count': sum(o.parent is None for o in objects),
          'roots': [r for r in rows if r['parent'] is None], 'objects': rows,
          'cameras': [r for r in rows if r['type'] == 'CAMERA'],
          'units': {'system': bpy.context.scene.unit_settings.system,
                    'scale_length': bpy.context.scene.unit_settings.scale_length}}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('INVENTORY=' + str(out))
