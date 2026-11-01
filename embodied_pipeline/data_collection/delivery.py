"""Offline delivery schemas, honest documentation and byte-level file manifests.

These helpers never call a model, launch WSL, load an embedding model or send a
robot command. Freeze a manifest only during offline preparation or after all
workers have stopped; the manifest is a snapshot, not a live progress file.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil

from .._legacy import ROOT


VERSION = "single_pass_delivery_v1"
MANIFEST_NAME = "file_manifest.json"
SKILL_IDS = ("navigate_to_object", "position_arm_for_object", "grasp_object",
             "lift_held_object", "place_object_on_surface")
HASH_SCHEMA = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
ID_SCHEMA = {"type": "string", "pattern": "^(scene200|scene201|scene202|d3_619)_[0-9]{8}$"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _wsl_path(path):
    value = str(path).replace("\\", "/")
    return "/mnt/" + value[0].lower() + value[2:] if re.match(r"^[A-Za-z]:/", value) else value


def dataset_schema():
    """Schema of catalogue rows and their later aggregate dataset extensions."""
    object_spec = {"type": "object"}
    properties = {
        "record_id": ID_SCHEMA, "sample_id": ID_SCHEMA,
        "scene_id": {"enum": ["scene200", "scene201", "scene202", "d3_619"]},
        "graph_version": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
        "graph_sha256": HASH_SCHEMA, "edges_sha256": HASH_SCHEMA,
        "instruction": {"type": "string", "minLength": 1},
        "instruction_language": {"const": "en"},
        "instruction_generated_at_utc": {"type": "string", "format": "date-time"},
        "instruction_generated_at_local": {"type": "string", "format": "date-time"},
        "source_scene_capture_time": {"type": ["string", "null"], "description": "Unknown stays null; not file mtime or generation time."},
        "task_family": {"enum": ["navigate", "single_transport", "pickup_hold", "protected_transport",
                                    "ordered_two_delivery", "ordered_three_delivery", "ordered_visit"]},
        "horizon": {"enum": ["short", "long"]},
        "meaningful_objective_count": {"type": "integer", "minimum": 1},
        "room_scope": {"enum": ["single_room", "cross_room", "unknown"]},
        "known_room_labels": {"type": "array", "items": {"type": "string"}},
        "room_provenance": object_spec,
        "grounded_object_ids": {"type": "array", "uniqueItems": True,
                                "items": {"type": "integer", "minimum": 0}},
        "initial_state": object_spec, "constraints": object_spec, "goal": object_spec,
        "available_skills": {"type": "object", "required": ["path", "sha256"],
                             "properties": {"path": {"type": "string"},
                                            "sha256": {"type": ["string", "null"]}}},
        "modality_resources": {"type": "array", "items": {"type": "object",
            "required": ["kind", "path", "sha256"], "properties": {
                "kind": {"type": "string"}, "path": {"type": "string"}, "sha256": HASH_SCHEMA}}},
        "source_permissions": object_spec, "reference": object_spec, "dedup": object_spec,
        "quality": {"type": "object", "required": ["status", "correctness_evaluated", "training_eligible"],
                    "properties": {"status": {"type": "string"}, "correctness_evaluated": {"const": False},
                                   "training_eligible": {"const": False}}},
        "modality_index": {"type": "string"}, "input_graph": {"type": "string"},
        "reference_artifact": {"type": "string"}, "source_type": {"type": "string"},
        "evaluation_status": {"type": "string"}, "collection_status": {"type": "string"},
        "model_run": {"type": ["string", "null"]}, "model_result": {"type": ["object", "null"]},
        "collection_times": {"type": "object", "properties": {
            "source_scene_capture_time": {"type": ["string", "null"]},
            "instruction_generated_at_utc": {"type": ["string", "null"]},
            "instruction_generated_at_local": {"type": ["string", "null"]},
            "started_at_utc": {"type": ["string", "null"]},
            "started_at_local": {"type": ["string", "null"]},
            "finished_at_utc": {"type": ["string", "null"]},
            "finished_at_local": {"type": ["string", "null"]},
            "duration_seconds": {"type": ["number", "null"]},
            "request_response_time_evidence": {"type": ["string", "null"]}}},
        "structural_quality": object_spec,
        "semantic_evaluation": {"type": "string"}, "valid_candidate": {"type": "null"},
    }
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Single-pass task candidate record",
            "description": "Each JSONL line is one independent synthetic task, not a frame, object, action or HTTP request. This schema does not establish semantic validity.",
            "type": "object", "required": ["record_id", "sample_id", "scene_id", "graph_version", "graph_sha256",
                "edges_sha256", "instruction", "instruction_language", "instruction_generated_at_utc",
                "source_scene_capture_time", "task_family", "horizon", "meaningful_objective_count",
                "room_scope", "grounded_object_ids", "initial_state", "constraints", "goal", "available_skills",
                "modality_resources", "source_permissions", "reference", "quality", "dedup"],
            "properties": properties, "additionalProperties": True,
            "$comment": "Record/scene matching, object membership, graph digests and dependency DAG require the supplied quality scripts in addition to JSON Schema."}


def reference_schema():
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "title": "Separate rule-authored reference, not independently verified human truth",
            "required": ["record_id", "source", "version", "steps", "dependencies", "goal",
                         "independent_of_model_output", "independently_human_verified"],
            "properties": {"record_id": ID_SCHEMA, "source": {"const": "rule_authored_version"},
                "version": {"type": "string"}, "independent_of_model_output": {"const": True},
                "independently_human_verified": {"const": False},
                "independent_reference_available": {"const": False},
                "steps": {"type": "array", "minItems": 1, "items": {"type": "object",
                    "required": ["step_id", "intent", "object_id", "destination_id", "depends_on"],
                    "properties": {"step_id": {"type": "string"}, "intent": {"enum": ["visit", "deliver", "hold_at"]},
                        "object_id": {"type": "integer", "minimum": 0},
                        "destination_id": {"type": ["integer", "null"]},
                        "depends_on": {"type": "array", "items": {"type": "string"}}}}},
                "dependencies": {"type": "array", "items": {"type": "array", "minItems": 2,
                    "maxItems": 2, "items": {"type": "string"}}}, "goal": {"type": "object"}},
            "additionalProperties": True}


def _versions(config):
    paths = {
        "catalog": "embodied_pipeline/data_collection/catalog.py",
        "batch": "embodied_pipeline/data_collection/batch.py",
        "worker": "embodied_pipeline/orchestration/task_collection_worker.py",
        "prompt": "embodied_pipeline/planning/intent_prompt.py",
        "alignment_adapter": "embodied_pipeline/planning/intent_alignment_worker.py",
        "quality": "embodied_pipeline/data_collection/quality.py",
        "resources": "embodied_pipeline/data_collection/resources.py",
        "delivery": "embodied_pipeline/data_collection/delivery.py",
    }
    return {"schema_version": VERSION, "requested_planning_model": config.get("model"),
        "base_url": config.get("base_url"), "provider_response_model": None,
        "provider_response_model_evidence": "Per-run planning_summary.response_evidence.response_models and actual saved HTTP response bodies; null before calls.",
        "model_weights_delivered": False, "post_training_performed": False,
        "prompt_arm": "intent_constraints_prompt", "matrix_contract": "typed5_schema",
        "process_constraints": "process_constraints_v1", "candidate_policy": "prerequisite_complete",
        "full_graph_decomposition_extension": True,
        "historical_experiment_equivalence": False,
        "historical_perception_model_note": "Preserve each source caption/generation manifest. Scene201 perception used qwen3.7-flash; scene200/202 and D3-619 evidence used qwen3.8-flash. These are not this batch's task-model output versions.",
        "prompt_raw_evidence": "cases/<scene>/tasks/<record_id>/model_run/llm_response_audit/*_request.json only after a request was prepared",
        "source_sha256": {name: {"path": rel, "sha256": _hash(ROOT / rel)} for name, rel in paths.items() if (ROOT / rel).is_file()}}


def _time_fields():
    return {"schema_version": VERSION, "unknown_timestamp_policy": "null; never infer scene capture date from filename or file modification time",
        "local_timezone": "UTC+08:00", "fields": {
            "source_scene_capture_time": "Original scene acquisition time; currently unknown=null. A render, graph export, download, or packaging date is not a substitute.",
            "instruction_generated_at_utc": "When synthetic task instructions were authored into this catalogue (UTC).",
            "instruction_generated_at_local": "The same task-authoring instant explicitly shown at UTC+08:00.",
            "collection_config.prepared_at_utc": "When this package/configuration was prepared; not a model call or observation.",
            "attempt_started.started_at_utc": "At-most-once local claim, before a task model request; claim does not prove the provider received anything.",
            "planning_summary.collection_started_at_utc": "Start of this task's collection attempt.",
            "planning_summary.collection_finished_at_utc": "End of collection attempt, including failed/partial outcomes.",
            "llm_response_audit.recorded_at": "Timestamp of this local audit event; request, HTTP response and transport failure events are different files.",
            "duration_seconds": "Measured local elapsed duration, not wall-clock acquisition time.",
            "dataset.collection_times": "Per-record aggregate of capture/authoring/start/end timing and the actual request/response audit directory; not-yet-run values remain null.",
            "file_manifest.generated_at_utc": "When byte hashes were frozen after writers stopped; not model inference time."}}


def write_delivery_docs(root: Path, config: dict):
    """Write offline package docs, actual symbolic skill definitions and schemas."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    library_source = ROOT / "outputs/quadruped_atomic_skill_library.v0.1.json"
    profiles_source = ROOT / "outputs/skill_matching_v1/skill_semantic_profiles.json"
    library, profiles = _read(library_source), _read(profiles_source)
    selected = copy.deepcopy(library)
    selected["skills"] = [s for s in selected["skills"] if s.get("skill_id") in SKILL_IDS]
    if {s["skill_id"] for s in selected["skills"]} != set(SKILL_IDS):
        raise ValueError("Collection skill definitions missing; no placeholder library will be fabricated")
    for skill in selected["skills"]:
        profile = profiles["profiles"][skill["skill_id"]]
        skill["description_en"] = profile["description_en"]
        skill["semantic_profile"] = copy.deepcopy(profile)
    selected["delivery_boundary"] = {
        "scope": "Actual five existing symbolic specifications and retrieval profiles used by this collection; dynamic per-task execution_contract is saved in model_run/execution_skill_library.json.",
        "source_sha256": _hash(library_source), "profile_source_sha256": _hash(profiles_source),
        "physical_capabilities_verified": False, "robot_execution_authorized": False}
    shared = root / "shared"
    shared.mkdir(exist_ok=True)
    shutil.copyfile(library_source, shared / "skill_library_source.json")
    shutil.copyfile(profiles_source, shared / "skill_semantic_profiles_source.json")
    _write(shared / "actual_collection_skill_library.json", selected)
    _write(root / "dataset.schema.json", dataset_schema())
    _write(root / "reference.schema.json", reference_schema())
    _write(shared / "model_versions.json", _versions(config))
    _write(shared / "timestamps.json", _time_fields())
    actual_tasks = list((root / "cases").glob("*/tasks/*/task.json"))
    actual_references = list((root / "cases").glob("*/tasks/*/reference.json"))
    raw_responses = list((root / "cases").glob("*/tasks/*/model_run/llm_response_audit/*_http_*.json"))
    plans = list((root / "cases").glob("*/tasks/*/model_run/engineering_plan.json"))
    from .plan_artifact import has_complete
    inventory = {"schema_version": VERSION, "prepared_at_utc": config.get("prepared_at_utc"),
        "requested_scene_count": len(config.get("scenes", {})),
        "requested_per_scene": config.get("requested_per_scene"),
        "actual_task_files": len(actual_tasks), "actual_reference_files": len(actual_references),
        "actual_http_response_files": len(raw_responses), "actual_engineering_plan_files": len(plans),
        "actual_complete_engineering_plans": sum(has_complete(p) for p in plans),
        "count_unit": "One independent task record. HTTP files, frames, steps and objects do not add records.",
        "prepared_does_not_mean_run": True, "valid_sample_count": None,
        "source_permissions": "Pending review; copying does not grant redistribution or training rights.",
        "locations": {
            "dataset": "dataset.jsonl", "config": "collection_config.json",
            "schemas": ["dataset.schema.json", "reference.schema.json"],
            "scene_graphs": "cases/<scene>/graph/scene_graph{,_edges}.json",
            "scene_catalogues": "cases/<scene>/tasks.jsonl",
            "tasks_and_references": "cases/<scene>/tasks/<record_id>/{task,reference}.json",
            "modality_index": "cases/<scene>/modality_index.json",
            "bundled_images": "cases/<scene>/modalities/images/",
            "model_outputs_only_after_run": "cases/<scene>/tasks/<record_id>/model_run/",
            "aggregate_quality": "quality_report.json", "code": "reproduction/", "file_hashes": MANIFEST_NAME}}
    inventory["locations"]["timestamps_index"] = "timestamps_index.jsonl"
    _write(root / "prepared_package_inventory.json", inventory)
    max_calls = config.get("max_full_collection_requests", 2400)
    repo, output = _wsl_path(ROOT), _wsl_path(root)
    text = f"""四场景任务推理结果包：先采集，后独立核验

一、这是什么
目标是 scene200、scene201、scene202、d3_619 各200条，共800条全新合成任务候选。
此前测试输出不混算本批记录。任务的步骤、帧、对象、三个模型阶段均不能另计一条。
这不是机器人采集，不是800条已验证正确数据，也不是已交付1万条。实际数量见
prepared_package_inventory.json、quality_report.json；未运行不得写成已运行。
合成标签不变，源资产和真实重建分别保留来源；数据许可/训练和再分发权限待确认。

二、样本与文件
record_id=scene_id加下划线加8位十进制编号，例如 scene200_00000001。
dataset.jsonl 每行一个任务，cases/<scene>/tasks.jsonl 是冻结指令清单。
tasks/<record_id>/task.json 含英文自然指令、初态、约束、目标、物体引用和来源。
同目录 reference.json 含规则编写的任务级参考步骤和依赖；它没有读取本次模型回答，
但与任务出自同一规则作者，不等于独立人工真值，independent_reference_available=false。
参考步骤不输入模型，不用同一生成模型复评冒充人工审核。
graph/ 保存完整原始节点/edges；modality_index.json 指向实际复制的图像与哈希。
每节点至多选择1张已有、有明确对象关联的图像，不冒充完整RGB-D/点云数据集。
shared/actual_collection_skill_library.json 是本批5项实际符号技能及检索描述；
shared/skill_library_source.json 保留完整原技能库。技能定义存在不证明真机已实现。
reproduction/、code_manifest.json 保存生成、运行、过滤/质检脚本和配置快照。

三、实验边界
本轮继续任务分解、物体绑定、技能绑定，尽可能生成工程JSON，但不执行机器人/仿真。
工程JSON状态是未评分的提案，不是可直接下发真机的命令。
这次是新扩展：完整场景图进入任务分解输入。不能宣称与历史仅指令分解实验完全相同。
不做room topology/连通图检索、不裁剪图、不虚构缺失空间关系。
scene201和D3-619房间归属保持未知；scene200/202仅沿用已有作者房间标注，
作者房间标注不等于已验证的感知真值，也不证明房间连通。
未观测机器人位姿、手持状态和可达性保持未知，不沿用前一任务失败/成功后的状态。
短程=1个任务级目标；长程=至少2个有意义且明确顺序的目标，不按原子动作数量凑长程。
参考的语义可行性、抓取能力、承载、目标空位等仍需后续人工分层抽核。

四、只做格式/引用质检，不比对答案
保留现有协议、对象引用、依赖环/前置引用、绑定一致性等工程检查；检查阻断不等于
给模型答案打错分。保留所有raw、失败与部分结果，不修正答案、不自动重试。
精确内容哈希用于查重复；目标/初态/步骤相同的近重复单列审核，不自动算有效。
统计分开报告：候选总数、已开始、真实HTTP响应、工程JSON、格式/引用无效、重复、
近重复待核、语义待核、已确认有效。未经语义核验有效数为未知，而不是把800全算有效。
不删除失败或重复原始文件；过滤前后数量/原因及人工审阅者另行记录。
source_scene_capture_time=null 表示采集时间未知，不可用生成时间代替。
完整时间字段说明见 shared/timestamps.json：指令生成、包准备、运行、请求/响应分别记录。
dataset.jsonl每条的collection_times可直接查看该任务生成/运行时间；未开始或未知字段保留null。
timestamps_index.jsonl按record_id独立汇总这些时间，便于交付核验。

五、资源授权和并发
准备文件只用CPU本地规则，不调用API。运行时每任务最多3次调用：分解、物体绑定、技能绑定。
本配置全量上限 {max_calls} 次（标准800条最多2400次）；这是调用次数上限，不是费用承诺。
新调用预算/模型服务入口和可用计算资源需本次负责人明确授权，旧预算不沿用。
4个场景进程并行，各场景内部顺序跑；本地embedding模型各进程常驻，先确认内存余量。
不自动下载模型/安装环境。API key 仅放用户终端环境变量，不写入配置或交付包。
同一任务一旦创建 model_run/ 即已占用一次机会：成功、失败、超时、进程中断均不重跑。
超时不能假定服务端没有计费；没有真实响应就保持缺失，绝不生成替代raw。
普通任务异常（含RuntimeError、解析/协议失败及收尾异常）记录后继续下一任务，不重试。
单场景启动/进程失败不停止其它场景；汇总更新失败也不取消正在运行的任务。
scene输入完整性失败、缺少共同环境等会阻止对应scene，不会伪造输入或绕过授权。
错误证据见 cases/<scene>/errors/、任务model_run/harness_error.json；不覆盖原响应。
断电、操作系统强杀或磁盘无法写入不能保证本进程继续；人工停止依旧生效。
运行期修订以 runtime_revision.json 和 runtime_revisions/ 追加保存，原代码快照和旧尝试保留。

六、由用户在WSL手动执行，助手不连接WSL
推荐开4个WSL终端，每个终端负责一个场景。每个终端先进入项目目录并指定同一结果包：
cd "{repo}"
export COLLECTION_OUTPUT_DIR="{output}"

确认本次资源/额度后，各终端分别运行下面对应的一行，先各跑2条：
bash embodied_pipeline/data_collection/run_scene.sh scene200 --approve-new-api-calls --limit-per-scene 2
bash embodied_pipeline/data_collection/run_scene.sh scene201 --approve-new-api-calls --limit-per-scene 2
bash embodied_pipeline/data_collection/run_scene.sh scene202 --approve-new-api-calls --limit-per-scene 2
bash embodied_pipeline/data_collection/run_scene.sh d3_619 --approve-new-api-calls --limit-per-scene 2

脚本在各窗口隐藏提示输入API key，实时显示进度并同时保存console_*.log。
4场景小批共最多8条/24次新调用；脚本单场景600次为保护上限，不会因此多跑任务。
先检查该小批格式与资源；每场景小批优先各选1条未开始的短程和长程任务，仍占同800编号，
不额外增加样本，不冒充覆盖全部任务类型的完整分层语义抽核。
放量前另按场景/长短程/任务类型检查代表性任务，样本量与审核者由负责人确认。
确认可以继续后，仍在对应4个窗口运行原命令，但删除 --limit-per-scene 2：
bash embodied_pipeline/data_collection/run_scene.sh scene200 --approve-new-api-calls
bash embodied_pipeline/data_collection/run_scene.sh scene201 --approve-new-api-calls
bash embodied_pipeline/data_collection/run_scene.sh scene202 --approve-new-api-calls
bash embodied_pipeline/data_collection/run_scene.sh d3_619 --approve-new-api-calls

已开始的成功/失败/未知结果均跳过，只处理剩余未开始任务。不要删除model_run来强制重试。
不同Python可在启动前export PIPELINE_PYTHON；embedding路径可追加
--embedding-python /实际环境/bin/python --embedding-model /实际本地模型目录。
不要同时启动下方统一管理入口和上方4窗口入口。

高级入口：单个主程序管理4场景进程。下面路径沿用原机环境，需要时自行替换。
cd "{repo}"
collection_root="{output}"
collection_python="/home/lewis/miniforge3/envs/conceptgraph/bin/python"
embedding_python="/home/lewis/miniforge3/envs/skill_embed/bin/python"
embedding_model="/mnt/d/NebulaAI/models/skill_embedding_minilm_v1"

先确认本次额度/资源后，在终端安全输入key：
read -rsp 'DashScope API key: ' DASHSCOPE_API_KEY; echo
export DASHSCOPE_API_KEY

先小批：每场景2条，共最多8条、24次新API调用；该命令本身就是显式同意本次调用。
"$collection_python" -m embodied_pipeline.data_collection.batch run \\
  --output "$collection_root" --limit-per-scene 2 \\
  --embedding-python "$embedding_python" --embedding-model "$embedding_model" \\
  --approve-new-api-calls --max-api-calls 24

小批后人工检查文件格式、资源引用与语义可行性（不要求先给模型答案评分），确认后再放量。
不删 model_run/。去掉limit后只续跑未开始的任务，不重复那8条：
"$collection_python" -m embodied_pipeline.data_collection.batch run \\
  --output "$collection_root" \\
  --embedding-python "$embedding_python" --embedding-model "$embedding_model" \\
  --approve-new-api-calls --max-api-calls {max_calls}

剩余预算实际按未开始任务*3检查；若8条均已占用，标准800条批次剩余最多2376次。
预算参数大于本次所需并不会增加任务或重试。先小批发现不足应停下报告，不盲目放量。
不要同时重复启动同一场景；worker.lock 冲突需人工确认进程是否仍在，不能直接删除锁重跑。

七、汇总与交付
运行原日志在 cases/<scene>/worker_*.log；每任务的model_run目录保存分解/绑定/技能
结果、工程JSON、generation_config.json、planning_summary.json和llm_response_audit/。
真实原响应是 *_http_*.json 中的 response_body（可能有凭证脱敏）；*_transport_*.json
只是传输失败，不是模型输出。parsed/normalized文件不是原始HTTP响应。
请求模型名见配置；服务端实际返回model、token使用量见原响应与response_evidence。
未调用前 provider_response_model=null，没有模型权重或post-training成果，不捏造模型版本。

重算汇总（不调用API）：
"$collection_python" -m embodied_pipeline.data_collection.batch summarize --output "$collection_root"

4个场景窗口均结束后，重算汇总并冻结清单（不调用API）：
"$collection_python" -m embodied_pipeline.data_collection.batch finalize --output "$collection_root"
核验当前包或复制后的包（把output换成拷贝位置）：
"$collection_python" -m embodied_pipeline.data_collection.batch verify --output "$collection_root"

仅在offline准备完成或所有worker结束后冻结文件清单；不要对正在写日志的包作最终哈希。
write_file_manifest(root) 写每个文件的SHA256与字节数，排除自身、*.pending和worker.lock；
若存在worker.lock会拒绝冻结。verify_file_manifest(root) 返回missing、changed、extra和count。
任何新增运行都会改变结果包；运行后须重做汇总/清单，再核验拷贝后的整包。
文件哈希一致只证明字节传输一致，不证明任务答案正确。接收方应再做权限、独立语义及资源核验。
"""
    (root / "README.txt").write_text(text, encoding="utf-8")
    return inventory


def _excluded(relative):
    return relative == MANIFEST_NAME or relative.endswith(".pending") or PurePosixPath(relative).name == "worker.lock"


def _files(root):
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Refusing symbolic link in delivery package: " + str(path))
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if not _excluded(relative):
                yield path, relative


def write_file_manifest(root):
    """Freeze byte hashes after writers stop; never delete or unlock anything."""
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Manifest root must be an existing directory")
    locks = sorted(root.rglob("worker.lock"))
    if locks:
        raise ValueError("Workers may still be active; refusing manifest freeze: " + str(locks[0]))
    files = []
    for path, relative in _files(root):
        before = path.stat()
        digest = _hash(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("File changed while hashing: " + relative)
        files.append({"path": relative, "sha256": digest, "bytes": after.st_size})
    manifest = {"schema_version": VERSION, "algorithm": "sha256",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "freeze_policy": "offline_or_all_workers_finished_no_live_log_writes",
                "excluded": [MANIFEST_NAME, "*.pending", "worker.lock"],
                "count": len(files), "total_bytes": sum(row["bytes"] for row in files), "files": files}
    _write(root / MANIFEST_NAME, manifest)
    return manifest


def verify_file_manifest(root):
    """Read-only integrity check, including traversal defense and new-file report."""
    root = Path(root).resolve(strict=True)
    result = {"status": "verified", "missing": [], "changed": [], "extra": [],
              "invalid_entries": [], "count": 0, "verified_count": 0}
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        result.update(status="manifest_missing", missing=[MANIFEST_NAME])
        return result
    manifest = _read(manifest_path)
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("Manifest files must be an array")
    expected = set()
    for index, row in enumerate(entries):
        result["count"] += 1
        raw = row.get("path") if isinstance(row, dict) else None
        if (not isinstance(raw, str) or not raw or "\\" in raw or ":" in raw or
                PurePosixPath(raw).is_absolute() or ".." in PurePosixPath(raw).parts or
                raw != PurePosixPath(raw).as_posix() or _excluded(raw) or raw in expected or
                not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))) or
                type(row.get("bytes")) is not int or row["bytes"] < 0):
            result["invalid_entries"].append(index)
            continue
        expected.add(raw)
        path = root.joinpath(*PurePosixPath(raw).parts)
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root):
            result["invalid_entries"].append(index)
            continue
        if not path.is_file():
            result["missing"].append(raw)
        elif path.stat().st_size != row["bytes"] or _hash(path) != row["sha256"]:
            result["changed"].append(raw)
        else:
            result["verified_count"] += 1
    result["extra"] = sorted(relative for _, relative in _files(root) if relative not in expected)
    if manifest.get("count") != len(entries):
        result["invalid_entries"].append("count_mismatch")
    if result["missing"] or result["changed"] or result["extra"] or result["invalid_entries"]:
        result["status"] = "integrity_difference"
    return result
