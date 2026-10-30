"""Build observation-only records from saved VLM replies; no new inference or graph commits."""
import argparse
import struct
from pathlib import Path
from update_memory import read, write, digest
from replay_vlm_rgbd import validate_detection, observation_record


def png_size(path):
    with path.open('rb') as stream:
        header = stream.read(24)
    if header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError('Invalid PNG header')
    return struct.unpack('>II', header[16:24])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    source = read(args.source_run / 'regression_summary.json')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for case in source['cases']:
        folder = args.source_run / case['case']
        report = read(folder / 'regression_summary.json')
        observations = []
        for frame in report['frames']:
            name = frame['frame']
            candidates = [folder / (name+'.occluded.png'), folder / (name+'.box.png')]
            image = next((f for f in candidates if f.exists()), None)
            if image is None:
                raise ValueError('Cannot determine audited input dimensions: '+name)
            width, height = png_size(image)
            answer = read(folder / (name+'.detection.json'))
            row = dict(frame)
            try:
                row['category_recognized'] = validate_detection(answer, width, height) is not None
            except ValueError:
                row['category_recognized'] = False
            observations.append(observation_record(row))
        entry = {'case': case['case'], 'observations': observations,
                 'category_recognized_views': sum(o['category_recognized'] for o in observations),
                 'category_only_views': sum(o['observation_state'] == 'category_only_position_unresolved' for o in observations),
                 'surface_accepted_views': sum(o['surface_evidence_accepted'] for o in observations),
                 'source_report_digest': digest(report)}
        results.append(entry)
        write(args.output_dir / (case['case']+'.observation_log.json'), entry)
    write(args.output_dir / 'observation_audit.json', {'status': 'audit_complete', 'source_run': str(args.source_run),
        'fresh_api_calls': False, 'graph_commit_performed': False, 'training_eligible': False,
        'scope': 'Reclassification of saved model claims, not independent identity verification or a new experiment', 'cases': results})
    for row in results:
        print(f"{row['case']}: category={row['category_recognized_views']} surface={row['surface_accepted_views']} category_only={row['category_only_views']}")


if __name__ == '__main__':
    main()
