"""Full-source geometry inventory and evidence rendering; never save the blend.

Run with Blender --background --disable-autoexec SOURCE --python-exit-code 2
--python render_blend.py -- --output-dir DEST [--inventory-only].
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import bpy
from mathutils import Vector

VERSION = 'full_scene_evidence_v1'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def bounds(meshes, scale):
    points = [o.matrix_world @ Vector(p) for o in meshes for p in o.bound_box]
    low = [min(p[i] for p in points) * scale for i in range(3)]
    high = [max(p[i] for p in points) * scale for i in range(3)]
    return low, high


def inventory():
    scene = bpy.context.scene
    scale = scene.unit_settings.scale_length
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid Blender unit scale')
    meshes = [o for o in scene.objects if o.type == 'MESH' and len(o.data.vertices)]
    grouped = {}
    for mesh in meshes:
        root = mesh
        while root.parent is not None:
            root = root.parent
        # Imported P_* furniture parts share an exact transform but have no
        # parent. Group only those names, never merge arbitrary nearby objects.
        key = ('imported_P_parts', tuple(round(float(v), 5) for row in root.matrix_world for v in row)) \
            if root.type == 'MESH' and root.name.startswith('P_') else ('root', root.name)
        grouped.setdefault(key, []).append(mesh)
    rows, mapping = [], {}
    for identity, (key, members) in enumerate(sorted(grouped.items(), key=lambda p: str(p[0])), 1):
        low, high = bounds(members, scale)
        extent = [b-a for a, b in zip(low, high)]
        # Do not silently invent a positive thickness for degenerate surfaces.
        valid = all(math.isfinite(v) and v > 0 for v in extent)
        rows.append({'id': identity, 'bbox_center': [(a+b)/2 for a, b in zip(low, high)],
                     'bbox_extent': extent, 'bounds_min': low, 'bounds_max': high,
                     'geometry_valid': valid, 'mesh_names': sorted(o.name for o in members),
                     'grouping': 'shared_transform_imported_parts' if key[0] == 'imported_P_parts'
                                 else 'blender_parent_hierarchy',
                     'instance_identity_verified': False,
                     'render_views': [], 'geometry_source': 'blend_world_aabb_oracle'})
        mapping[identity] = members
    assigned = [name for row in rows for name in row['mesh_names']]
    if len(assigned) != len(meshes) or len(set(assigned)) != len(meshes):
        raise ValueError('Full mesh coverage failed')
    return rows, mapping, scale


def relation_candidates(rows, tolerance=.10):
    """Broad on/in candidate gate only. AABB plausibility is NOT a relation fact."""
    pairs = []
    for a in rows:
        if not a['geometry_valid']:
            continue
        for b in rows:
            if a['id'] == b['id'] or not b['geometry_valid']:
                continue
            lo, hi, blo, bhi = a['bounds_min'], a['bounds_max'], b['bounds_min'], b['bounds_max']
            horizontal = all(min(hi[i], bhi[i]) - max(lo[i], blo[i]) > 0 for i in (0, 1))
            on = horizontal and abs(lo[2]-bhi[2]) <= tolerance
            contained = all(blo[i]-.01 <= lo[i] and hi[i] <= bhi[i]+.01 for i in range(3))
            strictly_smaller = math.prod(a['bbox_extent']) < .9 * math.prod(b['bbox_extent'])
            relations = (['on'] if on else []) + (['in'] if contained and strictly_smaller else [])
            if relations:
                pairs.append({'source': a['id'], 'target': b['id'], 'possible_relations': relations,
                              'candidate_only': True, 'render_views': []})
    return pairs


def setup_renderer(resolution, engine):
    scene = bpy.context.scene
    # Imported hierarchy/collection visibility must not silently drop assets.
    for collection in bpy.data.collections:
        collection.hide_render = False
    def enable(layer):
        layer.exclude = False
        for child in layer.children:
            enable(child)
    enable(bpy.context.view_layer.layer_collection)
    scene.render.engine = ('BLENDER_EEVEE' if bpy.app.version >= (5, 0, 0) else 'BLENDER_EEVEE_NEXT') \
        if engine == 'eevee' else 'BLENDER_WORKBENCH'
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.film_transparent = False
    # Disable saved compositor/sequencer effects; original file remains unchanged.
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.world = bpy.data.worlds.new('VLM_Audit_World')
    scene.world.use_nodes = True
    # Node names are localized in this user's Blender installation.
    tree = scene.world.node_tree
    background = next((n for n in tree.nodes if n.type=='BACKGROUND'),None)
    if background is None:
        background=tree.nodes.new('ShaderNodeBackground')
    output=next((n for n in tree.nodes if n.type=='OUTPUT_WORLD'),None)
    if output is None:
        output=tree.nodes.new('ShaderNodeOutputWorld')
    tree.links.new(background.outputs[0],output.inputs[0])
    background.inputs[0].default_value = (.75, .75, .75, 1)
    background.inputs[1].default_value = .8
    scene.view_settings.view_transform = 'Standard'
    if engine == 'workbench':
        scene.display.shading.light = 'STUDIO'
        scene.display.shading.color_type = 'TEXTURE'
        scene.display.shading.background_type = 'WORLD'
        scene.display.shading.show_shadows = True
        scene.display.shading.show_cavity = True
    for obj in scene.objects:
        if obj.type == 'LIGHT':
            obj.hide_render = True
    for index, rotation in enumerate(((.5, -.4, -.3), (-.4, .7, 2.2))):
        light = bpy.data.lights.new('VLM_Audit_Light_'+str(index), 'SUN')
        light.energy = 2.0 if index == 0 else 1.0
        obj = bpy.data.objects.new(light.name, light)
        scene.collection.objects.link(obj)
        obj.rotation_euler = rotation
    camera = bpy.data.objects.new('VLM_Audit_Camera', bpy.data.cameras.new('VLM_Audit_Camera'))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = 'ORTHO'
    camera.data.clip_start = .001
    camera.data.clip_end = 1000
    return camera


def render_view(out, name, ids, rows, mapping, scale, camera, angle, engine):
    scene = bpy.context.scene
    visible = {o.name for identity in ids for o in mapping[identity]}
    for obj in scene.objects:
        if obj.type == 'MESH':
            obj.hide_render = obj.name not in visible
    low = Vector([min(rows[i-1]['bounds_min'][k] for i in ids)/scale for k in range(3)])
    high = Vector([max(rows[i-1]['bounds_max'][k] for i in ids)/scale for k in range(3)])
    centre, diagonal = (low+high)/2, max((high-low).length, .01)
    direction = Vector((math.cos(angle), math.sin(angle), .65)).normalized()
    camera.location = centre + direction * diagonal * 2.5
    camera.rotation_euler = (centre-camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.ortho_scale = diagonal * 1.15
    bpy.context.view_layer.update()
    scene.render.filepath = str(out / name)
    destination = out / name
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.render.render(write_still=True)
    if destination.stat().st_size < 100:
        raise ValueError('Missing/invalid render '+str(destination))
    from bpy_extras.object_utils import world_to_camera_view
    projected = {}
    for identity in ids:
        r = rows[identity-1]
        lo, hi = r['bounds_min'], r['bounds_max']
        corners = [Vector((x/scale, y/scale, z/scale)) for x in (lo[0],hi[0])
                   for y in (lo[1],hi[1]) for z in (lo[2],hi[2])]
        uv = [world_to_camera_view(scene, camera, p) for p in corners]
        projected[str(identity)] = [min(p.x for p in uv), 1-max(p.y for p in uv),
                                    max(p.x for p in uv), 1-min(p.y for p in uv)]
    return {'path': name, 'sha256': sha(destination), 'camera_matrix_world': [list(r) for r in camera.matrix_world],
            'ortho_scale_blender_units': camera.data.ortho_scale, 'projected_aabb_uv': projected,
            'render_engine': engine, 'evidence_scope': 'isolated_asset' if len(ids)==1 else 'isolated_pair',
            'other_scene_objects_hidden': True, 'agent_reachable_camera': False}


def render_overview(out, rows, mapping, scale, camera, engine):
    """Global views and every broad flat floor patch; no filtering by task objects."""
    scene = bpy.context.scene
    for obj in scene.objects:
        if obj.type == 'MESH':
            obj.hide_render = False
    low = Vector([min(r['bounds_min'][i] for r in rows)/scale for i in range(3)])
    high = Vector([max(r['bounds_max'][i] for r in rows)/scale for i in range(3)])
    centre, diagonal = (low+high)/2, (high-low).length
    views = []
    specifications = [(f'overview/global_{k}.png', centre+Vector((math.cos(a),math.sin(a),1.2))*diagonal,
                       centre, diagonal*1.1, None) for k,a in enumerate((-.8,2.3))]
    for row in rows:
        size, c = row['bbox_extent'], Vector(row['bbox_center'])/scale
        if size[2] < .20 and size[0]*size[1] > 2 and c.z*scale < .30:
            # Overhead inside the source room; camera availability is oracle-only.
            specifications.append((f'overview/floor_{row["id"]}.png', c+Vector((0,0,2.7/scale)),
                                   c, max(size[0],size[1])*1.05/scale, row['id']))
    for name, position, target, zoom, floor_id in specifications:
        camera.location = position
        camera.rotation_euler = (target-position).to_track_quat('-Z','Y').to_euler()
        camera.data.ortho_scale = zoom
        destination = out/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        scene.render.filepath = str(destination)
        if not destination.is_file():
            bpy.ops.render.render(write_still=True)
        views.append({'path':name,'sha256':sha(destination),'camera_matrix_world':[list(r) for r in camera.matrix_world],
                      'ortho_scale_blender_units':zoom,'floor_candidate_id':floor_id,
                      'evidence_scope':'unaltered_full_scene','other_scene_objects_hidden':False,
                      'agent_reachable_camera':False,'render_engine':engine})
    return views


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--inventory-only', action='store_true')
    p.add_argument('--render-limit', type=int, default=0, help='QA only; positive limit never completes evidence')
    p.add_argument('--resolution', type=int, default=384)
    p.add_argument('--engine', choices=('eevee','workbench'), default='eevee')
    args = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(bpy.data.filepath)
    source_hash = sha(source)
    rows, mapping, scale = inventory()
    pairs = relation_candidates(rows)
    report = {'schema_version': '1.0', 'version': VERSION, 'status': 'inventory_only',
              'source_blend': str(source), 'source_sha256': source_hash,
              'render_script_sha256':sha(Path(__file__)),
              'coordinate_frame': 'source_blender_world_z_up_m', 'unit_scale': scale,
              'instance_count': len(rows), 'mesh_count': sum(len(x['mesh_names']) for x in rows),
              'full_source_mesh_assignment': True, 'inventory': rows, 'relation_candidates': pairs,
              'render_settings': {'resolution':args.resolution,'engine':args.engine},
              'expected_image_count': 2*len(rows)+2*len(pairs), 'completed_image_count': 0,
              'source_modified': False, 'perception_coverage_verified': False,
              'training_eligible': False,
              'limitations': ['All source meshes, including walls/floors, are inventoried.',
                  'Parent/P-part grouping is an asset segmentation heuristic, not semantic truth.',
                  'Isolated asset/pair views remove occlusion; not robot observations.',
                  'Only geometrically proposed on/in pairs are queried; absent edges remain unknown.',
                  'Source coordinates are simulator-oracle geometry, not VLM predictions.']}
    dest = out/'evidence_manifest.json'
    if dest.exists():
        old = json.loads(dest.read_text())
        if old['source_sha256'] != source_hash or old['version'] != VERSION or old['render_settings'] != report['render_settings']:
            raise ValueError('Existing evidence uses different source/settings; select a new output directory')
    save(dest, report)
    print('FULL_SCENE instances=%d meshes=%d relation_candidates=%d images=%d' %
          (len(rows),report['mesh_count'],len(pairs),report['expected_image_count']), flush=True)
    if args.inventory_only:
        return
    camera = setup_renderer(args.resolution, args.engine)
    report['overview_views'] = render_overview(out,rows,mapping,scale,camera,args.engine)
    save(dest,report)
    complete = 0
    for row in rows + pairs:
        identity = str(row['id']) if 'id' in row else str(row['source'])+'_'+str(row['target'])
        ids = [row['id']] if 'id' in row else [row['source'],row['target']]
        prefix = 'instances' if 'id' in row else 'pairs'
        for index, angle in enumerate((-.8, 2.3)):
            if args.render_limit and complete >= args.render_limit:
                report.update(status='qa_partial_evidence',completed_image_count=complete)
                save(dest, report)
                return
            row['render_views'].append(render_view(out, f'{prefix}/{identity}_v{index}.png', ids,
                                                   rows,mapping,scale,camera,angle,args.engine))
            complete += 1
        report.update(status='rendering',completed_image_count=complete)
        save(dest,report)
        print('RENDERED=%d/%d' % (complete,report['expected_image_count']), flush=True)
    if sha(source) != source_hash:
        raise ValueError('Source file changed during read-only rendering')
    report.update(status='evidence_complete',completed_image_count=complete)
    save(dest,report)


if __name__ == '__main__':
    main()
