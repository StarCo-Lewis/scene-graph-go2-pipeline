"""Validated multi-view captions with bounded format retries and raw evidence."""

import json
from pathlib import Path
import urllib.request
from uuid import uuid4

from .._legacy import load

ENTITY_TYPES = {"object", "furniture", "structure", "object_part", "unknown"}
INSTRUCTION = (
    "All images are views of the SAME segmented region. Return exactly ONE JSON object "
    "combining the evidence across views, never an array or one caption per image. "
    "Required keys: caption (nonempty string), possible_tags (array of strings), "
    "object_tag (nonempty string), entity_type (object, furniture, structure, object_part, "
    "or unknown), is_complete_object (boolean), parent_object_tag (string or null), "
    "confidence (number from 0 to 1). Generated text must be English. "
    "Do not infer hidden contents or physical affordances. If views are inconclusive, "
    "express uncertainty; do not invent a complete object."
)


def validate_caption(value, finish_reason=None):
    if finish_reason == "length":
        raise ValueError("Caption response was truncated")
    if not isinstance(value, dict):
        raise ValueError("Caption must be one JSON object, not " + type(value).__name__)
    required = {"caption", "possible_tags", "object_tag", "entity_type",
                "is_complete_object", "parent_object_tag", "confidence"}
    missing = required - value.keys()
    if missing:
        raise ValueError("Missing caption fields: " + ", ".join(sorted(missing)))
    if (not isinstance(value["entity_type"], str) or value["entity_type"] not in ENTITY_TYPES
            or type(value["is_complete_object"]) is not bool
            or type(value["confidence"]) not in (int, float) or not 0 <= value["confidence"] <= 1
            or not isinstance(value["possible_tags"], list)
            or not all(isinstance(v, str) for v in value["possible_tags"])
            or not all(isinstance(value[k], str) and value[k].strip() for k in ("caption", "object_tag"))
            or (value["parent_object_tag"] is not None and not isinstance(value["parent_object_tag"], str))):
        raise ValueError("Invalid caption field types or values")
    if any('\u4e00' <= c <= '\u9fff' for c in json.dumps(value, ensure_ascii=False)):
        raise ValueError("Caption response must use English text")
    return value


def request_caption(options, key, content, output_dir, *, transport=None, format_attempts=3):
    """Retry invalid model output, not exhausted transport errors or arbitrary bugs."""
    if format_attempts < 1:
        raise ValueError("format_attempts must be positive")
    audit_module = load("response_audit")
    transport = transport or load("llm_transport").request_json
    call_id, rows, feedback, http_attempts = uuid4().hex, [], "", 0
    ledger = Path(output_dir) / "caption_retry_audit" / (call_id + ".json")
    def save(status):
        audit_module.save(ledger, dict(schema_version="caption_retry_v1", call_id=call_id,
            status=status, attempts=rows,
            usage_scope="Per-response provider usage; absent/transport usage is unknown. "
                        "Adapter manifest usage covers accepted responses only; include this ledger for retry cost."))
    for attempt in range(1, format_attempts + 1):
        prompt = INSTRUCTION + ("\nPrevious response rejected: " + feedback +
            ". Generate a new response satisfying the same contract." if feedback else "")
        payload = dict(model=options.model, enable_thinking=False,
            response_format={"type": "json_object"}, max_completion_tokens=800,
            messages=[{"role": "user", "content": list(content) + [{"type": "text", "text": prompt}]}])
        req = urllib.request.Request(options.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        evidence = audit_module.ResponseAudit(output_dir, "real_scene_caption",
            context={"caption_call_id": call_id, "format_attempt": attempt}, request=req)
        row = dict(format_attempt=attempt, request_id_local=evidence.id, provider_usage=None)
        rows.append(row)
        print(f"[QWEN] Sending caption request; format attempt={attempt}/{format_attempts}, "
              "transport attempts=2, timeout=180s", flush=True)
        try:
            envelope = transport(req, timeout=180, attempts=2, audit=evidence)
        except Exception as exc:
            row.update(status="transport_failed", error_type=type(exc).__name__, error=str(exc))
            save("transport_failed")
            raise
        http_attempts += sum("_http_" in Path(p).name or "_transport_" in Path(p).name for p in evidence.files)
        row["provider_usage"] = envelope.get("usage") if isinstance(envelope, dict) else None
        text = None
        try:
            try:
                choice = envelope["choices"][0]
                text = choice["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("Missing caption response content") from exc
            if not isinstance(text, str):
                raise ValueError("Caption response content must be a JSON string")
            value = validate_caption(json.loads(text), choice.get("finish_reason"))
        except ValueError as exc:
            feedback = str(exc)
            evidence.parsed(text, "rejected", exc)
            row.update(status="format_rejected", error=feedback)
            save("retrying" if attempt < format_attempts else "format_exhausted")
            print(f"[QWEN] Caption format rejected: {feedback}; attempt={attempt}/{format_attempts}", flush=True)
            continue
        evidence.parsed(text, "validated")
        row["status"] = "validated"
        save("validated")
        return envelope, value, http_attempts
    raise ValueError(f"Caption format remained invalid after {format_attempts} attempts: {feedback}")
