"""Offline, full-inventory evidence verification; no VLM or simulator execution."""
import argparse
import json
import struct
from pathlib import Path
from generate_graph import read,save,digest


def validate(root):
    manifest=read(root/'evidence_manifest.json')
    if manifest['status']!='evidence_complete':
        raise ValueError('Full rendering has not completed')
    instances=manifest['inventory']
    ids={row['id'] for row in instances}
    if len(ids)!=len(instances) or len(ids)!=manifest['instance_count']:
        raise ValueError('Instance IDs/count mismatch')
    meshes=[name for row in instances for name in row['mesh_names']]
    if len(meshes)!=len(set(meshes)) or len(meshes)!=manifest['mesh_count']:
        raise ValueError('Mesh coverage mismatch')
    references=manifest['overview_views'][:]
    expected=manifest['expected_image_count']
    instance_pair_count=0
    for row in instances+manifest['relation_candidates']:
        if len(row['render_views'])!=2:
            raise ValueError('Two-view evidence missing')
        references.extend(row['render_views'])
        instance_pair_count+=2
        if 'source' in row and (row['source'] not in ids or row['target'] not in ids):
            raise ValueError('Invalid relation candidate endpoints')
    if instance_pair_count!=expected or manifest['completed_image_count']!=expected:
        raise ValueError('Incomplete coverage counter')
    for view in references:
        path=root/view['path']
        if not path.is_file() or digest(path)!=view['sha256']:
            raise ValueError('Missing/modified image: '+str(path))
        with path.open('rb') as f:
            header=f.read(24)
        if header[:8]!=b'\x89PNG\r\n\x1a\n':
            raise ValueError('Invalid PNG: '+str(path))
        if struct.unpack('>II',header[16:24])!=(manifest['render_settings']['resolution'],)*2:
            raise ValueError('Unexpected image dimensions')
    report={'status':'full_geometry_evidence_checked','source_sha256':manifest['source_sha256'],
            'evidence_manifest_sha256':digest(root/'evidence_manifest.json'),
            'instances':len(instances),'meshes':len(meshes),'relation_candidates':len(manifest['relation_candidates']),
            'instance_pair_images':instance_pair_count,'overview_images':len(manifest['overview_views']),
            'all_png_checksums_and_dimensions_valid':True,'geometry_degenerate_instances':[
                row['id'] for row in instances if not row['geometry_valid']],
            'vlm_generation_completed':(root/'scene/scene_graph.json').is_file(),
            'semantic_accuracy_verified':False,'simulation_executed':False,'training_eligible':False}
    save(root/'evidence_validation.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(validate(args.sample_dir.resolve()),ensure_ascii=False,indent=2))
