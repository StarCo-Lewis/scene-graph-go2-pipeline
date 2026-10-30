"""Local response evidence; no request headers, credentials or invented responses."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def category(error):
    text = str(error).lower()
    rules = [("transport_timeout", ("timeout", "timed out")),
             ("invalid_json", ("jsondecode", "expecting value", "expecting property", "extra data", "unterminated")),
             ("unknown_skill", ("unknown skill", "unsupported_skill", "not in skill", "only listed skill")),
             ("semantic_redundancy", ("redundant_lift", "semantic_redundancy")),
             ("relation_unverified", ("relation_unverified", "initial object relation", "relation_unknown")),
             ("object_binding_ambiguous", ("binding_ambiguous", "ambiguous")),
             ("object_binding_missing", ("no_object_can", "no object can", "missing_object", "nonexistent", "ungrounded")),
             ("runtime_precondition", ("proximity", "object_not_held", "preparation_required", "gripper_not_empty", "end state")),
             ("capability_claim", ("capability", "capabilities", "unavailable planning")),
             ("response_schema", ("role_id", "must be", "required_roles", "subtask", "schema", "ordered_step_ids", "truncated"))]
    return next((name for name, words in rules if any(word in text for word in words)), "other_validation")


def redactor(extra_secrets=()):
    secrets = set(v for v in extra_secrets if isinstance(v, str) and len(v) >= 8)
    for name, value in os.environ.items():
        if any(word in name.upper() for word in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")) and len(value) >= 8:
            secrets.add(value)
    def clean(value):
        if isinstance(value, str):
            for secret in sorted(secrets, key=len, reverse=True):
                value = value.replace(secret, "[REDACTED_CREDENTIAL]")
            return re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/-]+", "Bearer [REDACTED_CREDENTIAL]", value)
        if isinstance(value, dict):
            return {str(k): clean(v) for k,v in value.items()
                    if str(k).lower() not in ("authorization", "api_key", "headers", "password", "secret", "access_token")}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value
    return clean


def save(path, data, cleaner=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (cleaner or redactor())(data)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2)+"\n", encoding="ascii")


class ResponseAudit:
    def __init__(self, directory=None, stage="unknown", context=None, request=None):
        self.root = Path(directory)/"llm_response_audit" if directory is not None else None
        self.id = uuid4().hex
        self.stage, self.context = stage, context or {}
        self.files = []
        auth = request.get_header("Authorization", "") if request is not None else ""
        self.clean = redactor((auth.removeprefix("Bearer "),))
        if self.root is not None:
            payload = json.loads(request.data.decode()) if request is not None and request.data else {}
            # Only the model input is retained; URL, headers and authentication are excluded.
            safe_payload = {k: payload[k] for k in ("model", "temperature", "response_format", "messages") if k in payload}
            self.write("request", {"model_input": safe_payload})

    def write(self, suffix, data):
        if self.root is None:
            return
        path = self.root/f"{self.stage}_{self.id}_{suffix}.json"
        save(path, {"schema_version": "1.0", "request_id_local": self.id, "stage": self.stage,
                    "context": self.context, "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "training_eligible": False, **data}, self.clean)
        self.files.append(str(path))

    def received(self, attempt, raw, http_status=200):
        if isinstance(raw, bytes):
            content = raw.decode("utf-8", "replace")
        else:
            content = str(raw)
        self.write(f"http_{attempt}", {"event": "http_response", "http_attempt": attempt,
            "http_status": http_status, "response_received": True, "response_body": content,
            "sha256_before_credential_redaction": hashlib.sha256(content.encode()).hexdigest(),
            "credential_redaction_possible": True})

    def transport_error(self, attempt, error, http_status=None):
        self.write(f"transport_{attempt}", {"event": "transport_failure", "http_attempt": attempt,
            "http_status": http_status, "response_received": False,
            "error_type": type(error).__name__, "error": str(error), "category": category(error)})

    def parsed(self, content, status="parsed_not_validated", error=None):
        self.write("parse", {"event": "content_parse", "response_content": content,
            "status": status, "error": str(error) if error is not None else None,
            "category": category(error) if error is not None else None})


def validation(directory, stage, attempt, status, error=None, context=None, evidence=None):
    if directory is None:
        return
    path = Path(directory)/"llm_response_audit"/f"validation_{stage}_{uuid4().hex}.json"
    save(path, {"schema_version": "1.0", "event": "program_validation", "stage": stage,
        "validation_attempt": attempt, "context": context or {}, "status": status,
        "error": str(error) if error is not None else None,
        "category": category(error) if error is not None else None,
        "evidence": evidence, "training_eligible": False,
        "recorded_at": datetime.now(timezone.utc).isoformat()})


def response_text(response):
    raw = response.read()
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
