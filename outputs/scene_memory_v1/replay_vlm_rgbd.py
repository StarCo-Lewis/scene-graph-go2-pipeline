#!/usr/bin/env python3
"""RGB-only VLM -> SAM -> depth surface evidence. No semantic mask input or pose overwrite."""
import argparse
import base64
import io
import json
import logging
import math
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from update_memory import read, write, digest
from visual_fusion import project_pixel, fuse_frame

SCENE = Path('/mnt/d/NebulaAI/output/batches/scene201_vith_full_20260915_181750/seq_0001__hf_house_201')
LOG = logging.getLogger('vlm_rgbd_replay')


def occluded_width(width, fraction):
    if width <= 0 or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('Partial dropout requires positive width and fraction in (0,1)')
    return min(width, max(1, math.ceil(width*fraction)))


def observation_record(frame):
    recognized = frame.get('category_recognized') is True
    surface = frame.get('status') == 'accepted'
    return {'frame': frame['frame'],
            'observation_state': ('category_and_surface_evidence' if surface else
                                  'category_only_position_unresolved' if recognized else 'not_confirmed'),
            'category_recognized': recognized, 'surface_evidence_accepted': surface,
            'category_evidence_source': 'RGB_VLM_validated_category_box' if recognized else None,
            'stable_instance_identity_independently_verified': False,
            'position_update_allowed': False, 'delete_allowed': False,
            'reason': frame.get('reason'),
            'scope': 'Observation-only audit; not a graph event or complete-object pose'}


def validate_detection(answer, width, height):
    if not isinstance(answer, dict) or answer.get('visibility') not in ('visible', 'not_visible', 'uncertain'):
        raise ValueError('Invalid detection visibility')
    instances = answer.get('instances')
    if not isinstance(instances, list):
        raise ValueError('instances must be an array')
    if answer['visibility'] != 'visible':
        if instances:
            raise ValueError('Non-visible response cannot contain detections')
        return None
    if len(instances) != 1:
        raise ValueError('Ambiguous instance count; cannot bind a stable object ID')
    detection = instances[0]
    if not isinstance(detection, dict) or detection.get('label_en', '').strip().lower() not in ('kettle', 'electric kettle', 'water kettle'):
        raise ValueError('Detection label is not a kettle')
    box = detection.get('bbox_xyxy_1000')
    if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1000 for v in box):
        raise ValueError('Expected finite normalized 0..1000 xyxy box')
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Degenerate box')
    pixels = [x0*width/1000, y0*height/1000, x1*width/1000, y1*height/1000]
    if pixels[2]-pixels[0] < 4 or pixels[3]-pixels[1] < 4:
        raise ValueError('Object is too small for reliable segmentation')
    return pixels


def query_rgb(image, model, base_url, key):
    # Only RGB is sent; no node pose, oracle labels, mask, or suggested pixel location.
    from PIL import Image
    buffer = io.BytesIO()
    Image.open(image).convert('RGB').resize((1024, 1024)).save(buffer, format='PNG')
    payload = {'model': model, 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Inspect the image independently. Locate every visibly identifiable electric/water kettle. '
         'Do not invent an object because it is requested. If uncertain or occluded, report uncertain. '
         'Return JSON only: {"visibility":"visible|not_visible|uncertain","instances":[{"label_en":"electric kettle",'
         '"bbox_xyxy_1000":[left,top,right,bottom]}],"evidence_en":"short visual evidence"}. '
         'Coordinates must be normalized 0..1000 relative to image width and height. Include all visible kettles. '
         'Use an empty instances array unless visibility is visible. Do not output 3D coordinates.'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')}}]}],
        'response_format': {'type': 'json_object'}, 'max_tokens': 700}
    request = urllib.request.Request(base_url.rstrip('/')+'/chat/completions',
        data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'}, method='POST')
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'skill_matching_v1'))
    from match_skills import request_json_with_retry
    raw = request_json_with_retry(request, timeout=240, attempts=3)
    text = raw['choices'][0]['message']['content']
    if isinstance(text, list):
        text = ''.join(p.get('text', '') for p in text)
    return json.loads(text)


def position_report(centres):
    if len(centres) < 2:
        return {'status': 'insufficient_multiview_evidence', 'pose_update_allowed': False}
    centre = [sum(p[i] for p in centres)/len(centres) for i in range(3)]
    spread = max(math.dist(p, centre) for p in centres)
    return {'status': 'surface_evidence_consistent' if spread <= .15 else 'inconsistent_views',
        'observed_surface_center_cg': centre, 'max_view_spread_m': spread,
        'pose_update_allowed': False,
        'reason': 'Partial visible surface is not the complete object bbox centre; no position overwrite'}


def run(args):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from segment_anything import sam_model_registry, SamPredictor
    if not args.sam_checkpoint.is_file():
        raise ValueError('SAM checkpoint missing')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu explicitly')
    if not args.cached_detections and not os.environ.get('DASHSCOPE_API_KEY', '').strip():
        raise ValueError('DASHSCOPE_API_KEY required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout),
        logging.FileHandler(args.output_dir / 'replay.log', encoding='utf-8')])
    memory_dir = getattr(args, 'memory_dir', None) or args.visual_dir / 'fused_memory'
    translation_mode = getattr(args, 'translation_evidence', False)
    state = read(memory_dir / 'memory_state.json')
    state_digest = digest(state)
    aliases = {'kettle', 'electric kettle', 'water kettle', 'kitchen electric kettle'}
    known = [key for key, entity in state['entities'].items() if entity['active'] and aliases.intersection(
        str(label).strip().lower() for label in [entity['node'].get('object_tag', '')] + entity['node'].get('possible_tags', []))]
    if known != ['67']:
        raise ValueError('This pilot requires exactly one known kettle, stable ID 67; ambiguous maps are rejected')
    frames = sorted(args.visual_dir.glob('view_*.json'))[:args.max_views]
    if len(frames) < 2:
        raise ValueError('At least two RGB-D observations required')
    LOG.info('Loading SAM %s on %s before making API requests', args.sam_model_type, args.device)
    predictor = SamPredictor(sam_model_registry[args.sam_model_type](checkpoint=str(args.sam_checkpoint)).to(args.device))
    layer = {'schema_version': '1.0', 'coordinate_frame': 'conceptgraphs_world', 'voxel_m': .03,
             'association': 'RGB_VLM_box_SAM_mask_unique_known_category_not_oracle', 'frames': {}, 'objects': {}}
    centres, reports = [], []
    for metadata_path in frames:
        name = metadata_path.stem
        image_path = metadata_path.with_suffix('.png')
        image = np.asarray(Image.open(image_path).convert('RGB'))
        height, width = image.shape[:2]
        metadata = read(metadata_path)
        # The archive also contains semantic labels. Deliberately access only depth and RGB.
        with np.load(metadata_path.with_suffix('.npz'), allow_pickle=False) as archive:
            depth = archive['depth'].copy()
            if archive['rgb'].shape[:2] != (height, width) or depth.shape != (height, width):
                raise ValueError('RGB/depth dimensions mismatch')
        test_mask = getattr(args, 'test_occlusion', 'none')
        query_path = image_path
        if test_mask != 'none':
            image = image.copy()
            if test_mask == 'full':
                covered = np.ones((height, width), dtype=bool)
            else:
                prior_mask = np.asarray(Image.open(args.occlusion_fixture / (name+'.mask.png'))) > 0
                ys_fixture, xs_fixture = np.where(prior_mask)
                if not len(xs_fixture):
                    raise ValueError('Occlusion fixture mask is empty')
                covered = np.zeros((height, width), dtype=bool)
                left, right = int(xs_fixture.min()), int(xs_fixture.max())+1
                top, bottom = int(ys_fixture.min()), int(ys_fixture.max())+1
                fraction = getattr(args, 'occlusion_fraction', .5)
                covered[top:bottom, left:left+occluded_width(right-left, fraction)] = True
            modality = getattr(args, 'dropout_modality', 'joint')
            if modality in ('rgb', 'joint'):
                image[covered] = 96
            if modality in ('depth', 'joint'):
                depth[covered] = np.nan
            query_path = args.output_dir / (name+'.occluded.png')
            Image.fromarray(image).save(query_path)
            write(args.output_dir / (name+'.occlusion.json'), {
                'mode': test_mask, 'covered_pixels': int(covered.sum()),
                'dropout_modality': modality,
                'physical_occlusion_simulated': False,
                'target_bbox_width_fraction': getattr(args, 'occlusion_fraction', .5) if test_mask == 'partial' else None,
                'fixture_mask_used_only_for_partial_test_construction': test_mask == 'partial',
                'scope': 'Synthetic RGB-D input dropout, not a physical occluder'})
        LOG.info('Detecting kettle from RGB only: %s', name)
        answer = (read(args.cached_detections / (name + '.detection.json')) if args.cached_detections
                  else query_rgb(query_path, args.model, args.base_url, os.environ['DASHSCOPE_API_KEY']))
        write(args.output_dir / (name + '.detection.json'), answer)
        report = {'frame': name, 'status': 'rejected', 'semantic_archive_used_for_inference': False,
                  'category_recognized': False}
        try:
            box = validate_detection(answer, width, height)
            if box is None:
                report['reason'] = 'VLM did not confirm visible kettle'
                reports.append(report)
                continue
            report.update(category_recognized=True, recognized_category='electric kettle',
                          bbox_xyxy_pixels=box,
                          recognition_scope='VLM category claim; not independently verified instance identity')
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(box=np.asarray(box, dtype=np.float32), multimask_output=False)
            mask = masks[0].astype(bool)
            confidence = float(scores[0])
            # The box is a prompt, not an object mask. Require SAM and clip leakage.
            clipped = np.zeros_like(mask)
            x0, y0 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
            x1, y1 = min(width, math.ceil(box[2])), min(height, math.ceil(box[3]))
            clipped[y0:y1, x0:x1] = True
            mask &= clipped
            report.update(sam_quality_score=confidence, sam_mask_pixels=int(mask.sum()))
            if confidence < .8 or int(mask.sum()) < 32:
                raise ValueError('SAM quality/area insufficient; score is not a calibrated success probability')
            valid = mask & np.isfinite(depth) & (depth > .05) & (depth < 10)
            report.update(valid_depth_pixels=int(valid.sum()),
                          valid_depth_fraction=float(valid.sum()/mask.sum()))
            if int(valid.sum()) < 32 or valid.sum()/mask.sum() < .8:
                raise ValueError('Insufficient valid depth inside SAM mask')
            median = float(np.median(depth[valid]))
            mad = float(np.median(np.abs(depth[valid] - median)))
            valid &= np.abs(depth - median) <= max(.08, 3*mad)
            ys, xs = np.where(valid)
            if len(xs) < 32:
                raise ValueError('Depth consistency filter rejected mask')
            stride = max(1, len(xs)//4096)
            points = []
            for y, x in zip(ys[::stride], xs[::stride]):
                xyz = project_pixel(int(x), int(y), float(depth[y, x]), metadata['intrinsics_fx_fy_cx_cy'],
                    metadata['camera_rotation_matrix'], metadata['camera_position_habitat'])
                points.append(xyz + list(map(float, image[y, x, :3])))
            centre = np.median(np.asarray(points)[:, :3], axis=0).tolist()
            write(args.output_dir / (name + '.points.json'), {'coordinate_frame': 'conceptgraphs_world',
                'points_xyzrgb': points, 'surface_center_cg': centre,
                'association': 'VLM_SAM_unique_known_kettle', 'semantic_labels_used': False})
            frame_id = str(metadata_path.resolve()) + ':vlm_sam'
            if not translation_mode:
                layer = fuse_frame(layer, state, frame_id, {'67': points})
            accepted = translation_mode or frame_id in layer['objects'].get('67', {}).get('frame_ids', [])
            report.update(status='accepted' if accepted else 'geometry_conflict',
                bbox_xyxy_pixels=box, sam_quality_score=confidence, point_count=len(points),
                observed_surface_center_cg=centre, distance_to_memory_bbox_center_m=math.dist(centre, state['entities']['67']['node']['bbox_center']))
            if accepted:
                centres.append(centre)
            Image.fromarray((mask*255).astype(np.uint8)).save(args.output_dir / (name + '.mask.png'))
            preview = Image.fromarray(image)
            ImageDraw.Draw(preview).rectangle(box, outline='red', width=2)
            preview.save(args.output_dir / (name + '.box.png'))
        except ValueError as exc:
            report['reason'] = str(exc)
        reports.append(report)
        write(args.output_dir / 'frame_reports.json', reports)
    proposal = position_report(centres)
    observations = [observation_record(frame) for frame in reports]
    write(args.output_dir / 'semantic_observation_log.json', {
        'schema_version': '1.0', 'record_type': 'semantic_observation_audit',
        'source_memory': str(memory_dir), 'source_memory_digest': state_digest,
        'observations': observations, 'graph_commit_performed': False,
        'interpretation': 'Not confirmed does not mean deleted; recognized category does not establish an accurate position or identity'})
    status = 'non_oracle_visual_evidence_ready' if proposal['status'] == 'surface_evidence_consistent' else 'insufficient_visual_evidence'
    if translation_mode and len(centres) >= 2:
        status = 'translation_surface_evidence_ready'
    summary = {'status': status, 'model': args.model, 'frames': reports, 'position_evidence': proposal,
        'category_recognized_view_count': sum(o['category_recognized'] for o in observations),
        'category_only_view_count': sum(o['observation_state'] == 'category_only_position_unresolved' for o in observations),
        'semantic_observation_log': str(args.output_dir / 'semantic_observation_log.json'),
        'semantic_labels_used_for_inference': False, 'simulator_pose_used_for_identity': False,
        'identity_policy': 'RGB category matched to exactly one existing kettle; not general instance re-identification',
        'camera_pose_provenance': 'simulator_known_pose', 'depth_provenance': 'simulator_RGBD',
        'memory_geometry_used_as_consistency_gate': not translation_mode, 'memory_position_updated': False,
        'source_memory_unchanged': digest(read(memory_dir / 'memory_state.json')) == state_digest,
        'training_eligible': False, 'fresh_vlm_calls': not bool(args.cached_detections),
        'scope': 'Ungated surface candidates for translation registration; not a map update' if translation_mode else 'Known-kettle VLM/SAM association and partial-surface RGB-D fusion, not independent pose recovery'}
    if sys.platform.startswith('linux'):
        import resource
        summary['process_peak_rss_gib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1048576, 3)
    write(args.output_dir / 'visual_layer_vlm_sam.json', layer)
    write(args.output_dir / 'position_evidence.json', proposal)
    write(args.output_dir / 'regression_summary.json', summary)
    print(f"status={status} summary={args.output_dir / 'regression_summary.json'}", flush=True)
    return 0 if status in ('non_oracle_visual_evidence_ready', 'translation_surface_evidence_ready') and summary['source_memory_unchanged'] else 2


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--visual-dir', type=Path, default=SCENE / 'full_multiround_memory/scene201_20260926_150648_443503/round_2/visual')
    p.add_argument('--sam-checkpoint', type=Path, default=Path('/mnt/d/NebulaAI/models/segment_anything/sam_vit_h_4b8939.pth'))
    p.add_argument('--sam-model-type', choices=('vit_h', 'vit_l', 'vit_b'), default='vit_h')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--model', default='qwen3.8-flash')
    p.add_argument('--base-url', default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    p.add_argument('--max-views', type=int, choices=(2, 3, 4, 5, 9), default=3)
    p.add_argument('--test-occlusion', choices=('none', 'partial', 'full'), default='none')
    p.add_argument('--occlusion-fraction', type=float, choices=(.25, .5, .75), default=.5,
                   help='Fraction of prior target bbox width, not exact object mask area')
    p.add_argument('--dropout-modality', choices=('rgb', 'depth', 'joint'), default='joint',
                   help='Synthetic modality ablation; retained depth behind RGB fill is not physical occlusion')
    p.add_argument('--occlusion-fixture', type=Path, help='Saved SAM masks for constructing partial dropout only')
    p.add_argument('--cached-detections', type=Path, help='Replay saved detections without new API requests')
    p.add_argument('--memory-dir', type=Path, help='Known-object inventory checkpoint; no new-object fabrication')
    p.add_argument('--translation-evidence', action='store_true', help='Export surface candidates without stale-position gate; no automatic map update')
    p.add_argument('--output-dir', type=Path, default=SCENE / 'vlm_rgbd_memory_pilot' / datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f'))
    return p


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    if args.test_occlusion == 'partial' and args.occlusion_fixture is None:
        parser.error('--occlusion-fixture required for partial dropout')
    if args.test_occlusion != 'none' and args.cached_detections:
        parser.error('Occluded inputs require fresh VLM calls; cached detections are not valid')
    try:
        raise SystemExit(run(args))
    except Exception as exc:
        LOG.exception('VLM RGB-D replay failed')
        if isinstance(exc, FileExistsError):
            raise
        # Do not overwrite a previous run if the caller accidentally reused its path.
        if args.output_dir.exists() and (args.output_dir / 'regression_summary.json').exists():
            raise
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write(args.output_dir / 'regression_summary.json', {'status': 'error', 'error': str(exc), 'training_eligible': False})
        raise
