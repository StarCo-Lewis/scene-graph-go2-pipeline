# 2026-10-31：metadata 绑定补丁与符号规划小批

版本：pipeline-u87-symbolic-pilot-20261031-v1。本次仅更新规划 pipeline；没有部署狗端、修改导航参数、调用付费 API 或验证真机运动。

## 补丁验证与迁移

以 U87 提供方的三个 base SHA 核验当前源码，在独立源码副本上复现 RED：prepare 后修改 metadata.scene_id，旧版错误地接受了它。替换三文件后，五项 GREEN 检查通过；合入工作目录后再运行五项检查，也全部通过。外部操作防护记录表明没有 API、WSL、SSH、ROS、GPU 或凭据访问。

CLI prepare 与 GUI create_task 现在输出 real_scene_navigation_instruction_v2，绑定 graph、edges、metadata 三个准备时 SHA。planner 在输出创建前检查绑定。旧 instruction-v1 必须重新 prepare；已经打开的旧 GUI 必须重启并 reload 场景，不能给历史任务补写当前哈希来冒充原准备版本。

历史 real_scene_navigation_smoke_v1 路径未迁移，不宣称本补丁覆盖它。旧发布 manifest 保留；新源码包使用新的 manifest。三个 U87 源文件保持提供方 patched SHA，测试迁移和符号采集扩展另列。

验证证据在 outputs/metadata_u87_validation_20261031_v1：regression/red、regression/green、regression/green_current，以及 unit_tests_final.log。后者覆盖规划、GUI backend、全量候选、采集 worker 和符号小批，共 43 个离线测试；其中模型结果为测试 fixture，不是真实原响应。

## 新场景的输入边界

新增 TUM 单帧三节点，以及 ScanNet scene0488 的三帧联合图（一个场景，16 节点、10 个初批 active 节点）。保留 camera_optical / scannet_scene0488_unaligned_local_scene，不伪装为 D3-619 world frame。

新入口为 embodied_pipeline.data_collection.symbolic_pilot，复用原任务分解、物体绑定、技能绑定链路和五项符号技能。候选仍采用 full_inventory_no_cosine_v1；本次没有增加真实机器人技能能力。不会放宽真机 GUI 的坐标要求。

任务绑定三份图输入的准备时哈希；小批 manifest 绑定全部输入副本、模态索引、来源文件和 pipeline 源码。原始 graph/edges/metadata 不改写。空 edges 表示关系未知，不推测房间、on/in、起始位姿或坐标转换。

图像、深度、mask、点云和来源映射保留作审计材料，本轮模型收到的是文本图结构与 metadata，不是图像/VLM 实验。缺少 RGB 图不使 ScanNet XYZRGB 样本自动无效。来源采集时间按原始 metadata 保留；新记录的 prepare/run 时间不充当采集时间。

## 小批设计及运行

两个场景各 6 条英文任务，总共 12 条：指定 ID、唯一类别、两目标访问、三目标访问、类别歧义、目标缺失。ID 为 scene_id 加八位编号。参考步骤/依赖、终态目标和预期澄清单独保存在 references_not_model_input.json，不进入模型输入。

本机已经准备的目录为 outputs/symbolic_pilot_12_20261031_ready；v1/v2 是开发时的离线准备目录，没有 API 调用，不用于本次运行。

在原工程目录的 WSL 终端运行：

```bash
bash embodied_pipeline/data_collection/run_symbolic_pilot.sh
```

脚本不安装环境。默认 Python 为 /home/lewis/miniforge3/envs/conceptgraph/bin/python；换机器时用 PIPELINE_PYTHON 指定已经可运行 pipeline 的 Python。API key 在本地终端隐藏输入或使用已有 DASHSCOPE_API_KEY，不能写入代码/日志。模型固定 qwen3.8-flash，入口 https://dashscope.aliyuncs.com/compatible-mode/v1。

单条最多三个模型阶段请求，每阶段一次，无语义修复/自动重试；单条异常保存后继续。整个 results 目录独占创建，重复启动或已中断的批次不能自动重放。重新调用需要单独审核缺项、预算和新版本。若手动中断，已有原始日志保留，不把运行中的条目算完成。

预算约定 10 元。脚本限制 12 条/最多 36 个请求，**没有人民币计费硬限额**；应配置服务商限额并观察账单。token 单价、缓存、未知失败用量和实际账单尚未核实，不声称一定低于 10 元。此入口不能自动扩量。

## 审阅及扩量门槛

完成后检查 results/pilot_summary.json 和逐条目录中的 task_input、execution_contract、generation_config、input_audit、llm_response_audit、分解/绑定结果及 engineering_plan（仅成功生成时存在）。原响应、失败、模型返回版本和阶段时间均保留。

八条明确访问任务检查目标与顺序、依赖、末态、无无依据推断；四条歧义/缺失目标任务检查是否合理澄清或拒绝。传输错误不算合理拒绝，生成工程 JSON 不等于语义正确或真机成功。延续 12 分量表；未观察阶段单列，不自动给分。

先审阅全部 12 条，确认输入/协议问题已经解决、合理澄清正常，再讨论大批量。系统性错误先修复并另存版本，不能覆写失败原响应或只保留通过结果。

扩量预算使用本轮实际每阶段输入/输出 token、实际模型版本、账单单价及耗时；按图大小、任务长度分层估计，另外列重试/异常预算。两个小图的平均成本不能直接代表所有 100 个场景。现有包仅两份新增可交输入，待其他场景验收后确定数量。预算及并发额度确认后才开启扩量。
