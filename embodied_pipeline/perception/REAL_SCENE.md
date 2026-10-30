# Recorded real-scene input for the real robot

`real-scene` uses the installed ConceptGraphs adapters and a recorded RGB-D
sequence. It does not substitute the synthetic Scene200/202 fixtures.
Graph generation and import end at the real-robot planning input boundary.
No Habitat/physics simulation or symbolic action rollout is part of this entry.

## D3-619 input

The source is `D:/NebulaAI/input/D3-619_地图_v1_2026-10-21/D3-619_v1`.
It contains 483 undistorted 1280x960 RGB images, aligned uint16 millimetre depth,
body masks, intrinsics and OpenCV camera-to-world matrices. Source world axes
are x forward, y left, z up; units are metres. RGB-D covers only the observed
area, whereas the independent LiDAR map covers the room more broadly. Capture
dates are October 12/13, not the October 21 export date.

The adapter converts JPEG to lossless PNG and writes the original C2W matrices
without an OpenGL axis conversion. Body-mask pixels are zeroed in derived depth
only. Every source frame and every prepared artifact has an individual hash.
Source files are never edited. The `ai2thor` prepared layout is only the existing
adapter's file convention; it does not indicate simulated data.

The installed `streamlined_mapping.py` consumes `dataset.poses[idx]` directly.
This preserves the supplied world frame despite the dataset class also computing
relative poses. The runner checks that code path and records source hashes.

## Run manually in WSL

From any directory:

```bash
bash /mnt/c/Users/1/Documents/Codex/2026-09-15/https-chatgpt-com-s-cx-6aa9090c5f188191aeb739f2c2f6be8b/embodied_pipeline/perception/run_d3_619.sh
```

The launcher uses the existing `/home/lewis/miniforge3/envs/conceptgraph/bin/python`
and asks for `DASHSCOPE_API_KEY` without echo only if it is absent. It does not
read secret files or run WSL from Windows. It verifies and reuses
`outputs/d3_619_prepared_20261021/prepared` when present. All new run output stays
in this project's `outputs/real_scene_runs/d3_619_<timestamp>/`.

Stages: input verification -> class-agnostic SAM ViT-H + ViT-H-14 CLIP ->
ConceptGraphs overlap mapping -> English multi-view Qwen labels -> existing
upstream Qwen relation adapter on the all-node overlap MST -> full-precision scene export ->
pipeline input validation. Stride defaults to 1 (all 483 frames).

Logs are written to `<run>/logs/`; `run_summary.json` records the current stage.
API HTTP responses, usage and transport errors are retained under caption and
relation audit directories. A failed run preserves its intermediate outputs.
There is no automatic model replacement, semantic repair or full-run retry.
Each launch uses a new run and detection variant. Reusing a run output directory
is rejected. An 8 GiB GPU may need `--sam-model-type vit_b`; this is an explicit
experiment change, not an automatic fallback.

## Continue the October 21 failed caption stage

After WSL networking is restored, run manually:

```bash
bash /mnt/c/Users/1/Documents/Codex/2026-09-15/https-chatgpt-com-s-cx-6aa9090c5f188191aeb739f2c2f6be8b/embodied_pipeline/perception/resume_d3_619.sh
```

This prompts for the API key without echo if the current shell has none. It
checks successful detection/mapping records, prepared-input hashes, upstream
source hashes and the compressed map, then runs captions, relations and export
into a new `outputs/real_scene_runs/d3_619_semantics_<timestamp>/` directory.
Detections and mapping are reused. The old failed run and its request evidence
remain intact. The map hash is measured at continuation; the old run did not
record a map hash, so this is not retrospective proof of unchanged map contents.

This is a continuation from the original mapping checkpoint. By default another
launch starts fresh semantic requests. To reuse completed captions explicitly,
add `--reuse-captions-from <failed-semantic-continuation>/run_summary.json`.
Only successful records matching the map hash, original checkpoint hash, model,
endpoint and upstream source hashes are copied after schema/raw-response checks.
The old run is unchanged. Crop paths still refer to retained source files; do not
delete that run. `captions/caption_reuse_manifest.json` records provenance and
inherited evidence, which must not be counted as newly billed requests.
The general entry is `python -m embodied_pipeline real-scene resume
--from-summary <original-run>/run_summary.json`. It accepts only a terminal
failed caption-stage original run with successful mapping evidence.
Progress remains in the new `logs/captions.log`, whose path is printed at start.

Caption responses must be one combined JSON object for all views of the same
region. Invalid JSON/schema receives explicit feedback and at most three format
attempts, each with at most two transport attempts and a 180-second timeout.
Exhausted transport errors stop immediately; they do not start another format
cycle. Arrays are rejected, never silently reduced to their first item. All raw
responses and rejections are retained. `captions/caption_retry_audit/` records
provider usage for rejected and accepted responses; missing usage is unknown.
The upstream caption manifest counts accepted responses only, so it is not a
complete billing total when format retries occurred. HTTP transport details
remain in `llm_response_audit/`. Format validity does not prove semantic accuracy.

## Planning boundary

### Rebuild only the graph from the completed October 22 captions

```bash
bash embodied_pipeline/perception/rebuild_d3_619_graph.sh
```

Run this from the project root in WSL. It verifies the completed map hash, all
373 caption records against their raw responses, and upstream adapter hashes.
It reuses detection, mapping and all captions; only overlap/MST construction,
relation requests, export and input validation run. New output is written to
`outputs/real_scene_runs/d3_619_graph_<timestamp>/`; progress is in `logs/graph.log`.
The key is requested without echo if absent. No WSL or API is launched by Codex.

The original `build_qwen_conceptgraph_baseline.py` executes unchanged, including
its Qwen replacement prompt and retry policy. Raw requests/responses are audited
without changing their payload. `scene/upstream_export/` preserves the original
output. The planner-facing scene preserves every ID, caption, tag and edge;
there is no added completeness/type/confidence filter and no node `active` field.
For interface compatibility ONLY, AABB center/extent values are read from the
same source point cloud at full precision instead of the adapter's one-decimal
serialization. `geometry_precision_adapter.json` records both versions. True
zero dimensions cause an explicit failure, not node deletion or fabricated size.
Edges are copied byte-for-byte; they were inferred by the original adapter using
its rounded geometry, and precision adaptation does not re-judge those edges.

The prior `d3_619_semantics_20261022_003717_158342` graph is a historical extra-filter
variant (101 candidates out of 373), not the faithful all-node baseline. Its eight
edges are not reused. Old runs/captions remain read-only and must be retained.
Default full-run/resume exports now also use the upstream adapter; the old
filtered exporter is retained only as an explicitly named legacy regression path.

Generated files:

- `scene/scene_graph.json`: all mapped nodes, stable per-run IDs and full AABBs.
- `scene/scene_graph_edges.json`: active-node semantic/geometric hypotheses.
- `scene/map/scene_map_cfslam_pruned.pkl.gz`: stored map features with captions.
- `scene/scene_metadata.json`: world-frame and uncertainty provenance.
- `pipeline/pipeline_scene.json`: validated input paths, hashes and readiness.
- `pipeline/coordinate_contract.json`: original Z-up map axes and pending real-robot TF registration.

`embodied_pipeline.perception.load_real_scene(scene_dir)` loads and validates
the actual graph for downstream consumers. `real-scene import --scene-dir ...
--output-dir ...` validates an existing export without model calls.
Every upstream node is retained, including regions described as unknown or
incomplete. These descriptions remain in the source caption evidence and are
not converted to operation permissions. Missing/failed caption records stop
export instead of silently dropping nodes. Tiny positive extents are not rounded
to zero. Compatibility active counts mean retained nodes, not safe manipulation.

Only candidate MST pairs receive relation queries. Missing edges remain unknown;
this is not an exhaustive relation graph or a room-connectivity graph. A model
relation is not independently verified support/contact. The legacy binding
checker tests consistency with supplied edges, even if those edges are marked
as hypotheses; it does not perform physical validation.

No runtime state, empty-gripper assumption, user location or task is fabricated.
For later real-robot execution, supply current measured robot/held state,
applicable user context, a task, and the real-robot skill backend. Declare the
actual ROS map frame and current robot/sensor transforms into this recorded SLAM
map. The original x-forward/y-left/z-up convention is retained; no Y-up simulator
projection is generated or required. The separate `world_frames` helper belongs
to optional historical proxy compatibility and is not called by this entry.
OpenCV C2W input does not require an OpenGL flip. The LiDAR grid is registered by
the source metadata but is not converted to a navigation topology or claimed to
prove current robot reachability.

## Planning-only navigation smoke test

Run from WSL (the script locates the project, asks for a hidden API key if needed,
and makes fresh decomposition, object-binding and skill-matrix requests):

```bash
bash embodied_pipeline/perception/plan_d3_619_smoke.sh
```

Use `--preflight-only` to check and snapshot inputs without API or embedding calls.
The equivalent module entry is `python -m embodied_pipeline real-scene-plan`.
The English task is in `d3_619_navigation_smoke.json`: navigate near round table
179, without manipulating objects. Its graph and edge hashes pin the completed
`d3_619_graph_20261022_094742_334872` export. This does not regenerate the map.

Each launch creates `outputs/real_scene_runs/d3_619_plan_<timestamp>/`. The full
373-node, 91-edge scene is copied byte-for-byte, not replaced by a single-object
fixture. The existing five-skill inventory, English semantic profiles, cosine
retrieval and typed5 matrix protocol are reused. Original skill parameters,
including the optional navigation standoff preset, are retained without proxy
projection. The robot profile describes intended capabilities, not a connected
robot. The test does not use a deterministic navigation action compiler.

Main evidence: `task_decomposition.json`, `object_alignment.json`,
`skill_retrieval.json`, `action_codebook.json`, `llm_matrix_proposal.json`,
`decoded_matrix.json`, `skill_plan.json`, `static_review.json`, and
`task_output_check.json`. The task-specific check is applied AFTER generation;
it never inserts reference actions or repairs a failed model output.

`engineering_plan.json` wraps the unchanged skill plan with coordinate/source
references and pending interface requirements. It is a proposal envelope, NOT
a ROS command or an implemented robot API schema. `execution_ready` remains
false; the robot goal pose remains null. The object's bounding-box centre must
not be used directly as a collision-free base pose. Real localization/TF,
approach planning, stop tolerances, collision checks, skill interfaces and
execution feedback still need integration.

`planning_json_generated_not_executed` means the task-output and conditional
static checks passed, not that the robot reached the table. The static review
may inspect an optimized copy, but the exported plan always preserves the raw
decoded action sequence, including repetitions. Unknown physical preconditions
remain unknown. No proxy, Habitat, physical execution or memory commit is
invoked. Preflight and offline fixture tests are not live LLM successes.

## 仅导航的 ROS2 回放接入（首次现场联调）

入口：`python -m embodied_pipeline navigation-replay`。
这是新增的独立执行适配器，不修改历史计划，也不修改通用的 `RobotBackend`。
当前只接受一条 `navigate_to_object` 导航动作，并保留原动作编号。
含抓取、放置等操作或多个导航目标的计划会被整体拒绝，不会悄悄丢弃其中的动作。

提供的导航仓库中没有 TCP8888 网桥实现。本版直接使用 ROS2 Action
`/nav/navigate_to`，消息类型为 `nav_interfaces/action/NavigateTo`。
电脑需要可用的 ROS2 环境和编译后的对应接口。回放不会调用 LLM，也不需要 API key。

### 配置内容与坐标约定

配置文件：`embodied_pipeline/execution/d3_619_navigation_replay.json`。
通过哈希校验锁定已生成的工程计划、场景图和边文件；这不是重新生成计划。
尚未现场测量的位姿、未确认的限速和容差均保留为 `null`，防止使用猜测值驱动机器人。

| 配置项 | 含义与填写要求 |
| --- | --- |
| `start_pose` | 机器人实际起始位置与朝向，填写 `x`、`y`、`yaw` |
| `approach_pose` | 目标物体旁、机器人可安全停靠的位置与朝向，填写 `x`、`y`、`yaw` |
| `target_element_id` | 当前绑定的目标物体编号，本次为圆桌 `179` |
| `limits.pos_tol` | 导航位置容差，单位米 |
| `limits.yaw_tol` | 导航朝向容差，单位弧度 |
| `limits.timeout` | 导航任务超时时间，单位秒 |
| `limits.max_speed` | 导航速度上限，单位米/秒；本适配器要求大于零，不接受上游表示不限速的 `0` |
| `ros_environment` | ROS 通信配置，必须与实际运行环境一致 |
| `confirmations` | 人工检查项；完成对应检查后才设为 `true` |

注意：`approach_pose` 是圆桌旁的机器人停靠位姿，不是桌子的包围盒中心。
计划中的 `standoff_m=0.6` 会被记录，但不会被当作 `pos_tol`，也不会据此自动猜测停靠点。
现场应确认所选停靠点符合“停在圆桌附近”的任务要求。

所有位姿必须使用同一个 D3-619 场景坐标系：X 向前、Y 向左、Z 向上；
位置单位为米，`yaw` 使用弧度，向左转为正。`z`、`roll`、`pitch` 明确填为零。
适配器不会猜测坐标轴旋转、缩放或场景到机器人的变换。
如果现场导航地图使用不同坐标系，应先取得标定变换，再接入，不能直接混用坐标。
导航服务根据起点 `s_*` 建立场景系与里程计系的对应关系，适配器不重复进行这次变换。

所有容差、超时和限速必须填写现场确认的正数。人工检查项的含义如下：

| 检查项 | 设为 `true` 前需要确认 |
| --- | --- |
| `initial_pose_measured` | 机器人确实位于填写的起点，朝向也已核对 |
| `scene_frame_verified` | 起点、终点和导航服务使用的场景坐标约定一致 |
| `approach_pose_verified` | 圆桌旁停靠点有足够空间，符合任务要求 |
| `navigation_limits_verified` | 容差、限速和超时参数已经现场确认 |
| `operator_and_estop_ready` | 有现场操作人员看护，物理急停可用 |
| `server_alignment_policy_reviewed` | 已核对导航服务当前的坐标对齐状态和重对齐策略 |

仓库服务端默认的 `realign_on_mismatch=True` 可能在后续起点不一致时触发重新对齐；
它不是可以放进导航 Goal 的字段，需要与导航组核对服务端设置。
本客户端不会重置或重启服务端。重新启动客户端，不等于服务端重新建立了初始坐标系。
再次运行执行命令会发送新的真实导航请求，不是无副作用的重复检查。
未确认上一条指令的最终状态前，不要并发运行或重复发送。

### 今天可以做：离线检查（不需要 ROS2）

由你在 WSL 中执行以下命令；Codex 不启动 WSL 或机器人进程：

```bash
cd "/mnt/c/Users/1/Documents/Codex/2026-09-15/https-chatgpt-com-s-cx-6aa9090c5f188191aeb739f2c2f6be8b"
python3 -m embodied_pipeline navigation-replay
```

未填写的模板会返回退出码 `2`，状态为 `configuration_incomplete_not_executed`
（配置尚未补齐，未执行）。这是预期的安全检查结果，不是模型失败。
填写完整后，离线运行会返回 `request_prepared_not_executed`（请求已准备，未执行）。
这仅说明字段转换通过，不代表真实可达、定位正确、物理安全或 ROS2 已联通。

每次运行都会新建 `outputs/real_scene_runs/navigation_replay_<id>/`：

- `config_snapshot.json`：本次配置快照。
- `navigation_request.json`：源文件检查通过后生成的导航请求或待补齐项。
- `run_summary.json`：本次运行结果与错误信息。
- `events.jsonl`：实际连接执行时的事件和反馈日志。

服务返回的非有限数值（如 NaN）会记录为 `null`，表示未知，不会当作零。
这个入口不提交 memory 更新。

### 明天现场准备：ROS2 环境和接口

当前 WSL 会话没有找到 ROS。先用 `cat /etc/os-release` 确认系统版本，
再选择与操作系统、机器人兼容的 ROS 发行版，不要直接照搬仓库作者的 Conda 路径。
ROS 安装、DDS 通信和现场联调仍待完成。运行客户端时应使用与已加载 ROS 环境兼容的
Python，不能默认 ConceptGraphs 的 Conda 解释器也能加载 ROS。

安装 ROS 和 colcon 后，先加载所选发行版的 `/opt/ros/<distro>/setup.bash`
（将 `<distro>` 替换为实际发行版名称）。随后以下命令建立接口工作空间并编译消息，
不会运行机器人导航：

```bash
mkdir -p "$HOME/nav_replay_ws/src"
# 仅在目标不存在时复制；已有文件需要先检查，不直接覆盖。
if [ ! -e "$HOME/nav_replay_ws/src/nav_interfaces" ]; then
  cp -a /mnt/d/navvvvvvvvv-main/navvvvvvvvv-main/nav_interfaces "$HOME/nav_replay_ws/src/"
fi
cd "$HOME/nav_replay_ws"
colcon build --packages-select nav_interfaces
source install/setup.bash
# 以下来自仓库演示，现场需要与导航节点确认一致：
export ROS_DOMAIN_ID=7
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 interface show nav_interfaces/action/NavigateTo
ros2 action list -t
ros2 action info /nav/navigate_to
```

查询接口和发现 Action 服务本身不会让机器人运动。电脑与机器人需要兼容的 DDS 设置
和可用的网络连接。WSL 的 NAT、VPN 或组播可能需要现场调整，不应盲目关闭防火墙。
选择 CycloneDDS 时必须安装对应的 RMW。若现场使用其他 domain 或 RMW，
应同步修改配置文件和环境变量。

### 明确授权后：执行真实导航

只有在配置、定位、通行空间、服务端状态、现场看护和物理急停都检查完成后，
才在同一个已加载 ROS 环境的终端中，进入工程根目录执行：

```bash
python3 -m embodied_pipeline navigation-replay --execute --confirm-motion
```

这条命令会尝试驱动真实机器人。适配器发送核对过的 16 个字段，等待反馈和终态结果，
不会自动重试、重新规划或替换目标。只有 ROS 状态为 `SUCCEEDED`、`success=true`
且 `reason=arrived` 时，才记录“服务端报告到达”；这不是独立测量得到的真实到位结论。

超时或中断后，客户端会尝试取消导航。但收到取消确认不等于机器人已经停止。
如果没有收到终态结果，会明确记录“执行状态未知，需要操作人员检查”。
应按现场急停流程处理，并在确认状态后才能发送新指令。
按 Ctrl+C 导致 ROS 关闭，或网络断开时，取消请求可能无法送达。
关闭客户端不能替代物理急停。

离线测试命令：`python -m unittest embodied_pipeline.tests.test_navigation_replay`。
测试使用虚构位姿和模拟通信，不代表现场导航已经成功。
