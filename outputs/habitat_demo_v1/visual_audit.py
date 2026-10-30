"""Read-only sampled Habitat RGB + state audit. No control or perception feedback."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import struct
import zlib
from pathlib import Path
from proxy_safety import camera_pose, AuditCameraRejected

LOG = logging.getLogger(__name__)
SENSOR = "audit_rgb"


def add_arguments(parser):
    parser.add_argument("--visual-audit", action="store_true", help="Record sampled RGB and state; requires renderer")
    parser.add_argument("--audit-stride", type=int, default=1, help="Capture every N navigation steps (default every step)")
    parser.add_argument("--audit-max-frames", type=int, default=2000)


def configure_camera(hs, agent_cfg):
    spec = hs.CameraSensorSpec()
    spec.uuid, spec.sensor_type = SENSOR, hs.SensorType.COLOR
    spec.resolution, spec.position, spec.hfov = [360, 640], [0., 1.5, 0.], 90
    agent_cfg.sensor_specifications = [spec]


def compact_state(state):
    backend = state.get("backend", state)
    result = {key: copy.deepcopy(backend.get(key)) for key in
              ("robot_position", "robot_rotation_wxyz", "memory_version")}
    result.update({key: copy.deepcopy(state.get(key)) for key in ("held_object_id", "lifted")})
    result["objects"] = {str(key): {k: copy.deepcopy(obj.get(k)) for k in ("center", "com", "support_id")}
                         for key, obj in backend.get("objects", {}).items()}
    return result


def save_rgb(path, rgb):
    """Small lossless RGB PNG writer; no new Pillow/ffmpeg dependency."""
    height, width, channels = rgb.shape
    if channels != 3 or str(rgb.dtype) != "uint8":
        raise ValueError("Expected uint8 RGB frame")
    raw = rgb.tobytes(order="C")
    scanlines = b"".join(b"\0" + raw[y * width * 3:(y + 1) * width * 3] for y in range(height))
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))


def render_rgb(sim, backend=None, camera_report=None):
    """Move only the camera extrinsic, never the agent, objects or simulation clock."""
    import numpy as np
    import quaternion
    agent = sim.get_agent(0)
    original = agent.get_state()
    pose = copy.deepcopy(original)
    import habitat_sim
    ignore = {int(backend.instances[67].object_id)} if backend is not None else set()
    q = original.rotation
    camera = camera_pose(sim, habitat_sim, list(map(float, original.position)),
                         [float(q.real), *map(float, q.imag)], ignore)
    if camera_report is not None:
        camera_report.update(camera, ignored_own_kettle_ids=sorted(ignore))
    target, eye = np.asarray(camera["target"]), np.asarray(camera["eye"])
    backward = eye - target
    backward /= np.linalg.norm(backward)
    right = np.cross([0., 1., 0.], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    pose.sensor_states[SENSOR].position = eye
    pose.sensor_states[SENSOR].rotation = quaternion.from_rotation_matrix(np.column_stack((right, up, backward)))
    try:
        agent.set_state(pose, reset_sensors=False, infer_sensor_states=False)
        rgb = np.array(sim.get_sensor_observations()[SENSOR][..., :3], copy=True)
    finally:
        agent.set_state(original, reset_sensors=False, infer_sensor_states=False)
    return rgb


class VisualAudit:
    def __init__(self, args, summary):
        self.enabled = bool(getattr(args, "visual_audit", False))
        self.root = args.output_dir / "visual_audit"
        self.summary = summary
        self.stride = getattr(args, "audit_stride", 1)
        self.limit = getattr(args, "audit_max_frames", 2000)
        if self.stride <= 0 or self.limit <= 0:
            raise ValueError("Audit stride and frame budget must be positive")
        self.frames, self.context, self.errors = [], {}, []
        self.samples, self.truncated = 0, False
        self.executor = self.backend = None
        self.enabled_rgb = True

    def __enter__(self):
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=False)
            self.summary["visual_audit"] = {"status": "recording", "viewer": str(self.root / "index.html")}
        return self

    def bind(self, backend, executor):
        self.backend, self.executor = backend, executor

    def capture(self, phase, force=True, **extra):
        if not self.enabled or self.executor is None:
            return
        self.samples += 1
        if not force and self.samples % self.stride:
            return
        if len(self.frames) >= self.limit:
            self.truncated = True
            return
        before = self.executor.snapshot()
        entry = {"index": len(self.frames), **self.context, "phase": phase,
                 "state": compact_state(before), **extra}
        try:
            if not self.enabled_rgb:
                entry.update(image=None, render_error="RGB disabled after previous rendering error; state-only audit")
                self.frames.append(entry)
                return
            entry["camera"] = {}
            rgb = render_rgb(self.backend.sim, backend=self.backend, camera_report=entry["camera"])
            filename = f"frame_{len(self.frames):05d}.png"
            save_rgb(self.root / filename, rgb)
            entry["image"] = filename
        except Exception as exc:
            # Do not make an otherwise failed/successful action pass/fail for a camera error.
            entry["image"] = None
            entry["render_error"] = str(exc)
            self.errors.append({"index": entry["index"], "error": str(exc)})
            LOG.warning("Visual audit frame %s unavailable: %s", entry["index"], exc)
            # State-only audit remains available without repeatedly failing GPU reads.
            self.enabled_rgb = isinstance(exc, AuditCameraRejected)
        self.frames.append(entry)
        if self.executor.snapshot() != before:
            raise RuntimeError("Audit rendering changed runtime state; execution stopped")

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        report = {"schema_version": "1.0", "evidence_source": "live_sampled_simulator_readback",
                  "coordinate_frame": "habitat_world_y_up", "frames": self.frames,
                  "execution_error": str(exc) if exc is not None else None,
                  "execution_status": "failed" if exc else self.summary.get("status"),
                  "frame_limit_reached": self.truncated, "render_errors": self.errors,
                  "training_eligible": False, "physical_grasp_verified": False,
                  "playback": "Sampled frames, not wall-clock real-time video",
                  "camera_policy": "short_ray_checked_chase_or_robot_mounted_v2",
                  "navigation_sample_stride": self.stride,
                  "scope": "Proxy robot marker; actual RGB scene; assumed attach/lift; verified settling separately"}
        try:
            write_viewer(self.root, report)
            self.summary["visual_audit"].update(status="partial" if self.errors or self.truncated else "saved",
                frame_count=len(self.frames), frame_limit_reached=self.truncated,
                manifest=str(self.root / "audit.json"), render_error_count=len(self.errors))
            LOG.info("Visual audit viewer=%s frames=%s", self.root / "index.html", len(self.frames))
        except Exception as error:
            self.summary["visual_audit"].update(status="failed", error=str(error))
            LOG.exception("Saving visual audit failed; execution outcome remains independent")
        return False


def write_viewer(root, report):
    root.mkdir(parents=True, exist_ok=True)
    (root / "audit.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="ascii")
    payload = json.dumps(report, ensure_ascii=True).replace("<", "\\u003c")
    (root / "index.html").write_text(VIEWER.replace("__AUDIT_DATA__", payload), encoding="utf-8")


VIEWER = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Scene201 仿真可视化审计</title><style>
body{margin:24px;background:#111827;color:#e5e7eb;font:16px system-ui}h1{font-size:24px}
.warn{color:#fbbf24}.grid{display:grid;grid-template-columns:minmax(320px,640px) minmax(320px,600px);gap:20px}
img,canvas{width:100%;background:#1f2937;border-radius:8px}pre{white-space:pre-wrap;font-size:13px}
button,select{padding:8px;margin:6px}input{width:80%}.card{background:#1f2937;padding:12px;border-radius:8px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>
<h1>Scene201 仿真可视化审计</h1>
<p class="warn">任务级代理：蓝点代表导航机器人位置，不是真实机器狗模型；抓取/抬升为假定成功的代理操作。画面不证明夹爪接触、机械臂轨迹或真实抓取。</p>
<p id="scope"></p><p id="outcome"></p>
<button id="play">播放</button><button id="prev">上一帧</button><button id="next">下一帧</button>
<select id="speed"><option value="300">采样播放 3.3帧/秒</option><option value="100">10帧/秒</option><option value="700">1.4帧/秒</option></select>
<select id="actions"><option value="0">跳转动作</option></select><br><input id="slider" type="range" min="0" value="0"><span id="counter"></span>
<div class="grid"><div><img id="rgb" alt="实际采样 RGB"><p id="rgbnote"></p></div>
<div><canvas id="map" width="600" height="360"></canvas><p>俯视状态示意图（不是视觉重建）：蓝色=机器人，橙色=水壶67，绿色=边柜13。连线只是采样点之间的示意，不代表完整路径。</p></div></div>
<div class="card"><p id="label"></p><pre id="details"></pre></div>
<script>const audit=__AUDIT_DATA__;const frames=audit.frames;let i=0,timer=null;
const $=id=>document.getElementById(id);$('slider').max=Math.max(0,frames.length-1);
$('scope').textContent=audit.evidence_source==='historic_action_boundary_readback'?'历史日志：仅动作边界状态，没有原始RGB或连续轨迹。':'实际执行期间采样；忽略API等待时长，不按真实时间播放。新版相机检查局部遮挡；无法确认安全视角时不输出RGB。';
$('outcome').textContent='执行状态：'+audit.execution_status+'；画面/状态记录数：'+frames.length+(audit.frame_limit_reached?'；达到录制上限，记录不完整':'')+(audit.execution_error?'；停止原因：'+audit.execution_error:'');
const seen=new Set();frames.forEach((f,n)=>{const key=f.action_id||'round_'+f.round;if(!seen.has(key)){seen.add(key);const o=document.createElement('option');o.value=n;o.textContent=key+' '+(f.skill_id||f.phase);$('actions').append(o);}});
const points=frames.flatMap(f=>[f.state.robot_position,...Object.values(f.state.objects||{}).map(o=>o.center)]).filter(p=>p&&p.every(Number.isFinite));
const xs=points.map(p=>p[0]),zs=points.map(p=>p[2]);const minx=Math.min(0,...xs)-.7,maxx=Math.max(0,...xs)+.7,minz=Math.min(0,...zs)-.7,maxz=Math.max(0,...zs)+.7;
const project=p=>[30+(p[0]-minx)/(maxx-minx)*540,330-(p[2]-minz)/(maxz-minz)*300];
function map(f){const c=$('map').getContext('2d');c.clearRect(0,0,600,360);c.fillStyle='#9ca3af';c.font='13px system-ui';c.fillText('Habitat x/z (meters), y is height',12,18);
c.strokeStyle='#60a5fa';c.setLineDash(audit.evidence_source==='historic_action_boundary_readback'?[5,5]:[]);c.beginPath();frames.slice(0,i+1).forEach((r,n)=>{if(!r.state.robot_position)return;const p=project(r.state.robot_position);n?c.lineTo(...p):c.moveTo(...p);});c.stroke();c.setLineDash([]);
function dot(p,color,label,size){if(!p)return;const [x,y]=project(p);c.fillStyle=color;c.beginPath();c.arc(x,y,size,0,Math.PI*2);c.fill();c.fillText(label,x+9,y-8);}
dot(f.state.robot_position,'#60a5fa','Robot proxy',7);Object.entries(f.state.objects||{}).forEach(([id,o])=>dot(o.center,id==='67'?'#fb923c':'#34d399',id==='67'?'Kettle 67':'Sideboard 13',id==='67'?6:10));}
function show(){if(!frames.length){$('details').textContent='没有采样帧；请检查运行日志。';return;}const f=frames[i];$('slider').value=i;$('counter').textContent=(i+1)+' / '+frames.length;
$('rgb').style.display=f.image?'block':'none';if(f.image)$('rgb').src=f.image;$('rgbnote').textContent=f.image?'实际RGB：机器人没有实体外观，请同时看右侧代理轨迹。':'无RGB：'+(f.render_error||'历史动作边界日志');
$('label').textContent='轮次 '+(f.round??'-')+' | '+(f.action_id||'-')+' | '+(f.skill_id||'-')+' | '+f.phase;
$('details').textContent=JSON.stringify({instruction:f.instruction,state:f.state,status:f.action_status,camera:f.camera,physics_frame:f.physics_frame,motion_phase:f.motion_phase,held_state_pending_action_commit:f.held_state_pending_action_commit},null,2);map(f);}
function stop(){clearInterval(timer);timer=null;$('play').textContent='播放';}
$('play').onclick=()=>{if(timer)return stop();if(i===frames.length-1)i=0;timer=setInterval(()=>{if(i>=frames.length-1)return stop();i++;show();},Number($('speed').value));$('play').textContent='暂停';};
$('slider').oninput=()=>{i=Number($('slider').value);show();};$('actions').onchange=()=>{i=Number($('actions').value);show();};
$('prev').onclick=()=>{i=Math.max(0,i-1);show();};$('next').onclick=()=>{i=Math.min(frames.length-1,i+1);show();};$('speed').onchange=stop;show();</script></html>'''


def historic_report(summary):
    frames = []
    for row in summary.get("rounds", []):
        for action in row.get("actions", []):
            for phase in ("before", "after"):
                if phase not in action:
                    continue
                frames.append({"index": len(frames), "round": row["round"],
                    "instruction": row.get("instruction"), "action_id": f"round_{row['round']}_{action['step_id']}",
                    "skill_id": action["skill_id"], "phase": phase, "action_status": action.get("status"),
                    "state": compact_state(action[phase]), "image": None})
    return {"schema_version": "1.0", "evidence_source": "historic_action_boundary_readback",
            "execution_status": summary.get("status"), "execution_error": summary.get("error"),
            "frames": frames, "frame_limit_reached": False, "training_eligible": False,
            "physical_grasp_verified": False, "scope": "Boundary states only; interpolated lines are NOT observed navigation paths"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Create a boundary-only audit from an existing run, without API or Habitat")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists():
        p.error("Output directory already exists; historical results are never overwritten")
    data = json.loads((a.run_dir / "pipeline_summary.json").read_text(encoding="utf-8"))
    write_viewer(a.output_dir, historic_report(data))
    print(a.output_dir / "index.html")
