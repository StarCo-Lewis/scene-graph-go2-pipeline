"""Reuse only audited successes from the same map/model, without changing history."""

import json
from pathlib import Path
import shutil

from .caption_contract import validate_caption
from .real_rgbd import sha, write


def reuse_captions(summary_path, destination, checkpoint, model, base_url, source_hashes):
    source = Path(summary_path).resolve()
    destination = Path(destination).resolve()
    if destination == source.parent or source.parent in destination.parents or destination in source.parents:
        raise ValueError("Caption reuse output overlaps its source")
    report = json.loads(source.read_text(encoding="utf-8"))
    origin = report.get("resumed_from", {})
    if (report.get("schema_version") != "real_scene_semantic_continuation_v1"
            or report.get("status") != "failed_or_interrupted"
            or report.get("current_stage") != "captions"
            or report.get("map_unchanged") is not True
            or report.get("upstream_source_unchanged") is not True
            or any(origin.get(k) != checkpoint.get(k) for k in ("summary_sha256", "map_sha256", "map_file"))
            or report.get("semantic_model") != model
            or report.get("base_url", "").rstrip("/") != base_url.rstrip("/")
            or report.get("source_code_hashes") != source_hashes):
        raise ValueError("Caption cache checkpoint/model/source integrity mismatch")
    captions = source.parent / "captions"
    responses = {}
    for path in (captions / "llm_response_audit").glob("*_http_*.json"):
        event = json.loads(path.read_text(encoding="utf-8"))
        if event.get("http_status") != 200:
            continue
        envelope = json.loads(event["response_body"])
        if envelope.get("id"):
            responses[envelope["id"]] = (envelope, path)
    records, evidence = [], []
    for path in sorted((captions / "objects").glob("object_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "SUCCESS":
            continue
        index = record.get("id")
        if (type(index) is not int or index < 0 or path.name != f"object_{index:03d}.json"
                or record.get("semantic_schema_version") != 2 or record.get("model") != model):
            raise ValueError("Invalid successful caption cache record: " + str(path))
        value = validate_caption(record.get("qwen"))
        if record.get("captions") != [value["caption"]]:
            raise ValueError("Caption cache text mismatch: " + str(path))
        if record.get("request_id") not in responses:
            raise ValueError("Missing raw response evidence for caption: " + str(path))
        envelope, raw_path = responses[record["request_id"]]
        choice = envelope["choices"][0]
        if (validate_caption(json.loads(choice["message"]["content"]), choice.get("finish_reason")) != value
                or record.get("usage") != envelope.get("usage")):
            raise ValueError("Caption cache disagrees with raw response: " + str(path))
        views = record.get("selected_views")
        if not isinstance(views, list) or not views:
            raise ValueError("Missing caption views: " + str(path))
        crops = []
        for view in views:
            crop = Path(view["crop"])
            if not crop.is_file():
                raise ValueError("Missing retained caption crop: " + str(crop))
            crops.append(dict(path=str(crop), sha256=sha(crop)))
        records.append(path)
        evidence.append(dict(id=index, source_record=str(path), sha256=sha(path),
            raw_response=str(raw_path), raw_response_sha256=sha(raw_path), crops=crops))
    # Validate everything first. Never seed an unchecked schema-version-only cache.
    if destination.exists():
        raise FileExistsError("Caption destination must be new: " + str(destination))
    (destination / "objects").mkdir(parents=True)
    for path in records:
        shutil.copy2(path, destination / "objects" / path.name)
    # Retain rejected responses as well, distinguishing inherited evidence by manifest.
    inherited = []
    for name in ("llm_response_audit", "caption_retry_audit"):
        if (captions / name).is_dir():
            shutil.copytree(captions / name, destination / name)
            inherited.extend(str(p.relative_to(captions)) for p in (captions / name).rglob("*.json"))
    result = dict(schema_version="caption_reuse_v1", source_summary=str(source),
        source_summary_sha256=sha(source), reused_count=len(records), records=evidence,
        inherited_evidence=inherited,
        provenance="Records copied unchanged; crop paths refer to retained original files. "
                   "Inherited evidence/usage is historical, not newly billed usage.")
    write(destination / "caption_reuse_manifest.json", result)
    return result
