"""Explicit GUI-triggered planning, using the existing WSL pipeline, never a robot executor.

The secret travels over stdin, not argv or a task file. This module can be tested
with injected subprocesses without starting WSL or making an API request.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

from .field_backend import ROOT, read, sha, write, wsl_path
from .field_worker import validate_job

MAX_CONFIG_BYTES = 32768


def validate_api(model, base_url, key):
    if not isinstance(model, str) or not model.strip() or len(model) > 128 or any(c.isspace() for c in model):
        raise ValueError("Model must be a nonempty identifier without whitespace")
    u = urlsplit(base_url)
    if (u.scheme != "https" or not u.hostname or u.username or u.password or u.query or u.fragment
            or any(c.isspace() for c in base_url)):
        raise ValueError("API base URL must be HTTPS without credentials, query, fragment or whitespace")
    if not isinstance(key, str) or not key or any(c.isspace() for c in key):
        raise ValueError("Enter a nonempty API key without whitespace; do not save it in task JSON")


def launch_command(planning_python, distro=""):
    python = wsl_path(planning_python)
    if not python.startswith("/"):
        raise ValueError("Use an absolute WSL Python path")
    # No shell; all paths are separate argv items, including paths with spaces.
    bootstrap = ("import os,sys; os.chdir(sys.argv[1]); sys.path.insert(0,sys.argv[1]); "
                 "from embodied_pipeline.ui.api_planner import main; raise SystemExit(main(['--stdin-config']))")
    prefix = ["wsl.exe"]
    if distro:
        if distro.startswith("-") or any(c in distro for c in "\r\n\x00"):
            raise ValueError("Invalid WSL distribution name")
        prefix += ["--distribution", distro]
    return prefix + ["--exec", python, "-B", "-c", bootstrap, wsl_path(ROOT)]


def config_for(folder, model, base_url, api_key, embedding_python, embedding_model):
    validate_api(model, base_url, api_key)
    return {"schema_version": "field_api_planning_v1", "api_key": api_key,
            "model": model, "base_url": base_url.rstrip("/"),
            "job": {"schema_version": "field_planning_job_v1", "folder": folder.relative_to(ROOT).as_posix(),
                    "task_sha256": sha(folder / "task.json")},
            "embedding_python": wsl_path(embedding_python), "embedding_model": wsl_path(embedding_model)}


def public_config(config):
    return {k: config[k] for k in ("schema_version", "model", "base_url", "job", "embedding_python", "embedding_model")}


def launch_wsl(command, config):
    """Called only after operator approval; tests inject Popen, never run WSL."""
    payload = json.dumps(config, ensure_ascii=True).encode("utf-8")
    if len(payload) > MAX_CONFIG_BYTES:
        raise ValueError("Planning configuration too large")
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    output, _ = child.communicate(payload)
    return {"exit_code": child.returncode,
            "message": output.decode("utf-8", errors="replace").replace(config["api_key"], "[REDACTED]")[-4000:]}


def claim_folder(folder):
    with (folder / "planning_owner.lock").open("x", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "utc": datetime.now(timezone.utc).isoformat(),
                   "policy": "Never automatically replay a reserved planning run"}, f)


def run_config(config):
    if config.get("schema_version") != "field_api_planning_v1":
        raise ValueError("Invalid API planning configuration")
    validate_api(config["model"], config["base_url"], config["api_key"])
    folder = validate_job(config["job"])
    claim_folder(folder)
    key = config["api_key"]
    result = {"status": "planning_process_failed", "robot_executed": False}
    env = dict(os.environ, DASHSCOPE_API_KEY=key, PYTHONUNBUFFERED="1")
    command = [sys.executable, "-B", "-m", "embodied_pipeline", "real-scene-plan",
               "--task-json", str(folder / "task.json"), "--output-dir", str(folder / "planning"),
               "--model", config["model"], "--base-url", config["base_url"],
               "--embedding-python", config["embedding_python"], "--embedding-model", config["embedding_model"]]
    child = None
    try:
        with (folder / "worker.console.log").open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in child.stdout:
                log.write(line.replace(key, "[REDACTED]"))
                log.flush()
            code = child.wait()
        result.update(status="planning_process_finished", exit_code=code)
    except Exception as exc:
        result["error"] = (type(exc).__name__ + ": " + str(exc)).replace(key, "[REDACTED]")
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        code = 2
    finally:
        # Keep reservation and original outputs. No automatic retry after any failure.
        write(folder / "api_result.json", result)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdin-config", action="store_true", required=True)
    parser.parse_args(argv)
    if sys.platform != "linux":
        raise RuntimeError("This subprocess runs in the existing WSL planning environment")
    config = {}
    try:
        raw = sys.stdin.buffer.read(MAX_CONFIG_BYTES+1)
        if len(raw) > MAX_CONFIG_BYTES:
            raise ValueError("Planning configuration too large")
        config = json.loads(raw)
        if not isinstance(config, dict):
            raise ValueError("Planning configuration must be an object")
        return run_config(config)
    except Exception as exc:
        # Do not echo the input document, key or traceback into desktop logs.
        message = type(exc).__name__ + ": " + str(exc)
        key = config.get("api_key") if isinstance(config, dict) else None
        if isinstance(key, str) and key:
            message = message.replace(key, "[REDACTED]")
        print(message, file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
