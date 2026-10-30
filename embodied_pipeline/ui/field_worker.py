"""User-started WSL planning queue. Never imports or launches a robot executor."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from .field_backend import ROOT, QUEUE, read, sha, write


def validate_job(record):
    if record.get("schema_version") != "field_planning_job_v1":
        raise ValueError("Unsupported queue job schema")
    folder = (ROOT / record["folder"]).resolve()
    if folder.parent != (ROOT / "outputs/real_scene_runs").resolve() or not folder.name.startswith("field_ui_"):
        raise ValueError("Queue job must refer to an immediate field_ui_* run directory")
    task = folder / "task.json"
    if task.is_symlink() or folder.is_symlink() or sha(task) != record["task_sha256"]:
        raise ValueError("Task integrity check failed")
    if (folder / "planning").exists():
        raise ValueError("Planning output already exists; automatic replay is forbidden")
    return folder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-python", type=Path, default=Path("/home/lewis/miniforge3/envs/skill_embed/bin/python"))
    parser.add_argument("--embedding-model", type=Path, default=Path("/mnt/d/NebulaAI/models/skill_embedding_minilm_v1"))
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        raise RuntimeError("Start this worker manually in the existing WSL planning environment")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("Set DASHSCOPE_API_KEY in this terminal first; no request was made")
    if not args.embedding_python.is_file() or not args.embedding_model.is_dir():
        raise ValueError("Check the embedding interpreter and model directory before starting")
    QUEUE.mkdir(parents=True, exist_ok=True)
    lock = QUEUE / "worker.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    stop = threading.Event()
    state = {"status": "ready", "pid": os.getpid(), "current_task": None}
    child = None
    def heartbeat():
        while not stop.is_set():
            snapshot = {**state, "utc": datetime.now(timezone.utc).isoformat()}
            tmp = QUEUE / "worker_status.pending"
            try:
                tmp.write_text(json.dumps(snapshot), encoding="utf-8")
                tmp.replace(QUEUE / "worker_status.json")
            except OSError:
                pass  # Do not pretend online if heartbeat cannot be written.
            stop.wait(1)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    print("Planning worker ready. GUI submissions may incur API charges. No robot execution.", flush=True)
    try:
        while True:
            jobs = sorted(QUEUE.glob("*.job.json"))
            if not jobs:
                time.sleep(.5)
                continue
            for job in jobs:
                claimed = job.with_name(job.name.replace(".job.json", ".claimed.json"))
                job.rename(claimed)
                result_path = job.with_name(job.name.replace(".job.json", ".result.json"))
                state.update(status="planning", current_task=job.name)
                result = {"status": "planning_worker_failed", "robot_executed": False}
                try:
                    folder = validate_job(read(claimed))
                    from .api_planner import claim_folder
                    claim_folder(folder)
                    command = [sys.executable, "-B", "-m", "embodied_pipeline", "real-scene-plan",
                               "--task-json", str(folder / "task.json"), "--output-dir", str(folder / "planning"),
                               "--embedding-python", str(args.embedding_python), "--embedding-model", str(args.embedding_model)]
                    print("START " + folder.name, flush=True)
                    with (folder / "worker.console.log").open("x", encoding="utf-8") as log:
                        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                        code = child.wait()
                    child = None
                    result.update(status="planning_process_finished", exit_code=code, folder=str(folder))
                    print("END " + folder.name + " exit_code=" + str(code), flush=True)
                except Exception as exc:
                    result["error"] = type(exc).__name__ + ": " + str(exc)
                finally:
                    write(result_path, result)
                    state.update(status="ready", current_task=None)
    except KeyboardInterrupt:
        print("Worker interrupted; no automatic task replay. Inspect claimed tasks before retrying.", flush=True)
        return 130
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        state["status"] = "stopped"
        stop.set()
        thread.join(timeout=2)
        # Remove only this worker's lock; preserve claims/results for audit.
        if lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()
        (QUEUE / "worker_status.json").write_text(json.dumps({**state, "utc": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
