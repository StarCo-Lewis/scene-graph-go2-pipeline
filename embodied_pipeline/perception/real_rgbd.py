"""Audit recorded Replica RGB-D and prepare the existing CG adapter layout."""

import hashlib
import json
import math
import re
from pathlib import Path

FRAME = re.compile(r"frame(\d+)\.jpg\Z")
DEPTH = re.compile(r"depth(\d+)\.png\Z")
MASK = re.compile(r"mask(\d+)\.png\Z")
WORLD = "d3_619_slam_world_x_forward_y_left_z_up_m"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="ascii")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def camera_config(path):
    """Parse this recorded numeric-only Replica camera format; reject ambiguity."""
    text = Path(path).read_text(encoding="utf-8-sig")
    camera = {}
    dataset = None
    for line in text.splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        key, separator, value = clean.partition(":")
        if not separator:
            raise ValueError("Invalid camera configuration line")
        value = value.strip()
        if key == "dataset_name":
            if dataset is not None or value != "replica":
                raise ValueError("Expected one Replica dataset definition")
            dataset = value
        elif key == "camera_params" and not value:
            continue
        else:
            if key not in ("image_height", "image_width", "fx", "fy", "cx", "cy", "png_depth_scale", "crop_edge") or key in camera:
                raise ValueError("Unknown or duplicate camera field: " + key)
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Non-finite camera parameter")
            camera[key] = number
    required = {"image_height", "image_width", "fx", "fy", "cx", "cy", "png_depth_scale"}
    if dataset != "replica" or not required <= camera.keys():
        raise ValueError("Missing Replica camera parameters")
    for key in ("image_height", "image_width"):
        if camera[key] <= 0 or not camera[key].is_integer():
            raise ValueError("Invalid image dimensions")
        camera[key] = int(camera[key])
    if min(camera[k] for k in ("fx", "fy", "png_depth_scale")) <= 0 or camera.get("crop_edge", 0) != 0:
        raise ValueError("Invalid calibration or unsupported crop")
    if not (0 <= camera["cx"] < camera["image_width"] and 0 <= camera["cy"] < camera["image_height"]):
        raise ValueError("Principal point is outside the image")
    camera["crop_edge"] = 0
    return dict(dataset_name=dataset, camera_params=camera)


def indexed(results, pattern):
    found = {}
    for path in results.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            index = int(match[1])
            if index in found:
                raise ValueError("Duplicate numeric frame index")
            found[index] = path
    return found


def inspect_input(source):
    import numpy as np
    source = Path(source).resolve()
    rgbd = source / "rgbd_conceptgraphs"
    config = camera_config(rgbd / "replica_config.yaml")
    color, depth, masks = (indexed(rgbd / "results", p) for p in (FRAME, DEPTH, MASK))
    ids = sorted(color)
    if not ids or set(depth) != set(ids) or (masks and set(masks) != set(ids)):
        raise ValueError("RGB/depth/body-mask frame indices do not match")
    if ids != list(range(len(ids))):
        raise ValueError("Replica frames must be consecutive starting at zero")
    poses = np.loadtxt(rgbd / "traj.txt", dtype=np.float64, ndmin=2)
    if poses.shape != (len(ids), 16) or not np.isfinite(poses).all():
        raise ValueError("Expected one finite 4x4 camera-to-world matrix per RGB-D frame")
    poses = poses.reshape(-1, 4, 4)
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid homogeneous pose row")
    rotations = poses[:, :3, :3]
    error = float(np.max(np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3))))
    determinants = np.linalg.det(rotations)
    if error > 1e-4 or np.max(np.abs(determinants - 1)) > 1e-4:
        raise ValueError("Pose rotation is not a proper rigid transform")
    manifest = json.loads((rgbd / "manifest.json").read_text(encoding="utf-8"))
    if len(manifest["frames"]) != len(ids) or [f["index"] for f in manifest["frames"]] != ids:
        raise ValueError("Manifest indices do not match RGB-D")
    return source, rgbd, config, color, depth, masks, poses, error


def audit(source, output, *, prepare=False):
    import numpy as np
    from PIL import Image
    source, rgbd, config, colors, depths, masks, poses, error = inspect_input(source)
    output = Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be outside the read-only source dataset")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version="real_rgbd_input_v1", status="auditing", source=str(source),
        frame_count=len(colors), coordinate_frame=WORLD, camera_frame="OpenCV_x_right_y_down_z_forward",
        unit="metre", camera=config["camera_params"], pose_conversion="none; original OpenCV C2W retained",
        max_rotation_orthogonality_error=error, body_mask_policy="zero masked depth in derived dataset only",
        coverage="RGB-D coverage only; LiDAR full-room coverage does not add semantic observations",
        source_capture_dates=["2026-10-12", "2026-10-13"], live_robot_state_available=False,
        source_files={}, frames=[], api_calls=0, conceptgraphs_executed=False,
        simulation_executed=False, physical_grasp_verified=False, training_eligible=False)
    for name in ("README.md", "rgbd_conceptgraphs/replica_config.yaml", "rgbd_conceptgraphs/traj.txt",
                 "rgbd_conceptgraphs/traj_tum.txt", "rgbd_conceptgraphs/manifest.json",
                 "rgbd_conceptgraphs/alignment_check.json", "grid/D3-619.yaml", "grid/D3-619.pgm"):
        path = source / name
        if path.is_file():
            report["source_files"][name] = dict(sha256=sha(path), bytes=path.stat().st_size)
    if prepare:
        for name in ("color", "depth", "pose"):
            (output / "prepared/dataset" / name).mkdir(parents=True)
        # The adapter directory format is ai2thor, not the physical source.
        write(output / "prepared/dataconfig.yaml", dict(config, dataset_name="ai2thor"))
    width, height = config["camera_params"]["image_width"], config["camera_params"]["image_height"]
    try:
        for i in sorted(colors):
            with Image.open(colors[i]) as image:
                if image.size != (width, height) or image.mode != "RGB":
                    raise ValueError(f"Unexpected RGB format in frame {i}")
                image.load()
                if prepare:
                    image.save(output / f"prepared/dataset/color/{i:06d}.png")
            with Image.open(depths[i]) as image:
                depth = np.asarray(image).copy()
                if image.size != (width, height) or depth.dtype != np.uint16:
                    raise ValueError(f"Expected 16-bit depth in frame {i}")
            masked = 0
            if masks:
                with Image.open(masks[i]) as image:
                    mask = np.asarray(image)
                    if mask.shape != depth.shape or not np.isin(mask, (0, 255)).all():
                        raise ValueError(f"Invalid body mask in frame {i}")
                    masked = int(np.count_nonzero(depth[mask == 255]))
                    depth[mask == 255] = 0
            record = dict(index=i, rgb_sha256=sha(colors[i]), depth_sha256=sha(depths[i]),
                body_mask_sha256=sha(masks[i]) if masks else None,
                valid_depth_fraction_after_body_mask=float(np.count_nonzero(depth) / depth.size),
                valid_body_depth_pixels_removed=masked,
                max_valid_depth_m=float(depth.max() / config["camera_params"]["png_depth_scale"]))
            if prepare:
                destination = output / "prepared/dataset"
                Image.fromarray(depth).save(destination / f"depth/{i:06d}.png")
                np.savetxt(destination / f"pose/{i:06d}.txt", poses[i], fmt="%.12f")
                record["prepared_files_sha256"] = {f"{kind}/{i:06d}.{extension}":
                    sha(destination / f"{kind}/{i:06d}.{extension}")
                    for kind, extension in (("color", "png"), ("depth", "png"), ("pose", "txt"))}
            report["frames"].append(record)
            if i % 50 == 0 or i == len(colors) - 1:
                print(f"[RGBD AUDIT] {i + 1}/{len(colors)}", flush=True)
        report["status"] = "prepared_not_mapped" if prepare else "input_audit_passed"
        report["valid_depth_fraction_range"] = [min(f["valid_depth_fraction_after_body_mask"] for f in report["frames"]),
                                                max(f["valid_depth_fraction_after_body_mask"] for f in report["frames"])]
        report["total_valid_body_depth_pixels_removed"] = sum(f["valid_body_depth_pixels_removed"] for f in report["frames"])
    except BaseException as exc:
        report.update(status="input_audit_failed", error=str(exc))
        raise
    finally:
        write(output / "input_audit.json", report)
    return report


def validate_prepared(prepared):
    prepared = Path(prepared).resolve()
    report = json.loads((prepared.parent / "input_audit.json").read_text(encoding="utf-8"))
    if (report.get("status") != "prepared_not_mapped" or report.get("coordinate_frame") != WORLD
            or len(report["frames"]) != report["frame_count"]):
        raise ValueError("Prepared input does not have a completed real-RGBD audit")
    config = json.loads((prepared / "dataconfig.yaml").read_text(encoding="utf-8"))
    if config != dict(dataset_name="ai2thor", camera_params=report["camera"]):
        raise ValueError("Prepared calibration differs from the audited camera")
    expected = set()
    for row in report["frames"]:
        files = row.get("prepared_files_sha256", {})
        wanted = {f"{kind}/{row['index']:06d}.{extension}"
                  for kind, extension in (("color", "png"), ("depth", "png"), ("pose", "txt"))}
        if set(files) != wanted or expected.intersection(wanted):
            raise ValueError("Incomplete or repeated prepared frame inventory")
        for name, digest in files.items():
            path = prepared / "dataset" / name
            if not path.is_file() or sha(path) != digest:
                raise ValueError("Prepared frame modified: " + name)
        expected.update(wanted)
    actual = {f"{kind}/{p.name}" for kind in ("color", "depth", "pose")
              for p in (prepared / "dataset" / kind).iterdir() if p.is_file()}
    if actual != expected:
        raise ValueError("Prepared frame count differs from the audit")
    return report
