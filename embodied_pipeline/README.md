# 模块化规划工程：兼容重组版

## 狗端系统时间同步（2026-10-30）

脚本已保存为 `embodied_pipeline/deployment/dog_sync_time.sh`。在电脑的 **Git Bash** 中、从工程根目录执行：

```bash
bash embodied_pipeline/deployment/dog_sync_time.sh
```

这替代手册中的旧命令 `bash /d/capo_live/dog_sync_time.sh`。脚本通过 SSH 连接 `unitree@192.168.123.18`，按提示输入 SSH／sudo 密码；不再依赖原电脑的 `askpass.sh` 或脚本内置密码。首次连接需核实 SSH 主机指纹。脚本只同步系统时间，不启动导航或发送运动目标。下载目录中的原始脚本未改写。

从任意 Git Bash 目录启动当前工作区的完整命令：

```bash
bash "/c/Users/1/Documents/Codex/2026-09-15/https-chatgpt-com-s-cx-6aa9090c5f188191aeb739f2c2f6be8b/embodied_pipeline/deployment/dog_sync_time.sh"
```

本目录新增七个业务模块、流程调度层和共享接口契约。旧工程仍在 `outputs/` 原位，不移动、不覆盖。此次不调整prompt、模型、输入筛选、矩阵协议、余弦策略、校验规则、执行语义或失败后的状态提交规则。

## 目录与职责

```text
embodied_pipeline/
├── perception/          场景输入与空间关系接口；后续接主动探索
├── planning/            任务分解、对象绑定、现有持有对象闭包
├── instruction_skills/  技能检索、技能绑定、动作提案与码表
├── validation/          协议、引用、关系、状态等现有校验接口
├── execution/           现有符号代理适配器；真机接口占位
├── errors/              已有错误类型、分类和无提案结果；不增加恢复
├── memory/              状态/历史提交及日志存储接口
├── orchestration/       提案事务、逐轮流程、批次与任务集调度
├── contracts/           模块职责、输入输出接口和测试替换接缝
├── tests/               无网络的结构与行为一致性测试
├── _legacy.py           加载原位旧实现并检查同名模块冲突
├── introspection.py     工程清单、旧文件哈希核验
└── legacy_manifest.json 重组前309个旧文件的哈希；不是冻结引擎副本
```

默认流程：读取场景和当前状态 → 规划（分解/对象绑定）→ 技能动作提案 → 原有协议校验 → 符号代理 → memory提交。校验仍在原先各阶段发生；不新增整轮语义闸门。错误仍按原先边界拒绝、停止动作或终止批次，不增设统一重试。

## 模块如何复用旧工程

| 新模块 | 主要复用的旧实现 |
| --- | --- |
| perception | `run_contextual_ambiguity.scene_fixture`、`scene_input_interface`、旧 `prepare_feedback` |
| planning | `plan_live_round.decompose_and_bind`、`align_objects`、`llm_graph_binding`、`state_aware_planning` |
| instruction_skills | `skill_retrieval`、`action_matrix`、`match_skills` |
| validation | `matrix_type_contract`、`structural_matrix`、`runtime_goal_protocol`、`strict_prompt_v2`等 |
| execution | `dialogue_execution_feedback.execute_proxy`、`SerialExecutor`、`SymbolicBackend` |
| errors | 原有异常类及 `response_audit.category`；无提案结果保持原格式 |
| memory | 原JSON读写与审计工具；反馈状态/历史提交事务整理到 `commit_feedback` |

适配导出的函数仍是旧函数对象，不复制或重新实现技能、解码器、提示词、对象绑定和执行算法。提案事务、逐轮循环及批次调度在新目录中按原流程组织；旧循环保留作为行为对照和旧入口。

这是兼容性重组，不是彻底移除旧文件之间的所有依赖。特别是：

- 输入准备仍委托旧 `prepare_feedback`，从而保留输入、日志和实验profile的原样组织。
- 分解与对象绑定仍通过旧的联合子进程入口运行；没有强行拆成更多模型请求。
- 私有测试诊断仍在全部回合处理后运行，不能修正计划或memory。
- validated V2模式明确委托旧实现；本次重点整理的是execution_feedback路径。

## 记忆边界

`memory.commit_feedback`集中管理反馈提交事务：从代理的resulting_state构造候选状态，推进原有memory_version，构造成功动作历史及所有动作结果，校验公开对话数据，再通过pending文件替换一并提交状态和历史。

为了保持原先写入失败时的行为，调度层在事务完成后接纳候选状态/历史，并通过memory模块保存原有辅助文件。它不生成第二套状态，也不根据目标成功与否重置状态。

必须保留：

- 提案无效的回合仍记录失败和版本，物体状态不凭空变化。
- 动作失败保留此前成功前缀及失败时已发生的副作用，不回滚到理想状态。
- 当前状态仍是 `symbolic_hypothesis_not_observation`，不是物理观测。
- `scene_memory_v1/update_memory.py`是另一套需要已验证观测证据的地图更新机制，不能拿它替代本次符号反馈提交。
- 请求/响应和校验日志仍由旧审计器写入原位置；日志记录不等于物体状态提交。

## 使用方式

在工程根目录运行（不是用户home目录）：

```powershell
cd "C:\Users\1\Documents\Codex\2026-09-15\https-chatgpt-com-s-cx-6aa9090c5f188191aeb739f2c2f6be8b"
python -m embodied_pipeline inspect
python -m embodied_pipeline verify-legacy
python -m embodied_pipeline matrix-followup --help
```

WSL中对应根目录为：

```bash
cd "/mnt/c/Users/1/Documents/Codex/2026-09-15/https-chatgpt-com-s-cx-6aa9090c5f188191aeb739f2c2f6be8b"
python -m embodied_pipeline inspect
```

只准备输入、不会调用模型的预检：

```bash
python -m embodied_pipeline matrix-followup \
  --preflight-only --matrix-contract typed5_schema \
  --output-dir outputs/my_new_modular_preflight
```

输出目录必须是新的。预检只准备每条任务链第一轮输入，不处理74轮，不评估LLM正确率。

不带 `--preflight-only` 的 `dialogue` 或 `matrix-followup` 是明确的正式实验入口，会沿用原来的模型调用、API key与本地embedding要求。本次重组没有启动这类调用。原shell脚本和原Python入口均继续可用，且仍使用旧流程；只有新包入口使用模块化调度。

旧入口默认参数保持不变：矩阵任务默认仍可能是typed9。若要五位协议，显式使用 `--matrix-contract typed5_schema`，没有静默改默认值。

新CLI在运行后额外保存 `architecture_manifest.json`，记录新模块源文件哈希和旧文件完整性。此文件不进入模型上下文，不改变旧实验summary格式。

## 离线验证

```bash
python -m unittest embodied_pipeline.tests.test_compatibility -v
```

测试内容包括：

- 七模块适配器确实指向旧函数；导入不触发模型调用；真机占位不能声称执行。
- 同输入、同离线返回下，新旧动作请求payload、全部相关JSON工件及错误状态一致。
- 成功、协议拒绝、基础设施错误、输入状态被意外修改、失败前缀、审计失败、历史校验失败等路径。
- 两场景预检，原16轮对话及11链74轮followup的计划分母保持不变。
- 保存的A04/A05/A06/A08仍按原规则拒绝；A09重复理由不被自动修复。
- 保存的A12动作提案在符号代理中回放，反馈与原执行文件一致，包括已携带鼠标移动的失败前缀。

新旧对照仅忽略耗时、审计时间、临时运行根路径和审计文件名UUID；在对照中固定运行身份作用域。状态、编号、哈希、动作顺序、失败前缀、接口参数和目标不因比较而被修正。

离线fixture或历史代理回放不是新的LLM实验，不计入任务通过率，也不是Habitat或真机执行。

## 尚未加入的功能

- 不构建/推断房间连通图，不做房间级或对象级检索，不裁剪完整输入。
- 不优化或改写prompt，不改编号协议，不引入分层规划。
- 不增加语义重规划、澄清、动作修复或新的传输重试策略；保留旧传输重试。
- 不接主动探索、额外感知或真机控制。
- 不实现Oracle答案或post-training。接口允许显式替换proposer/executor供测试；后续独立Oracle-D/B仍需单独设计并标记干预，不能把测试替代输出算作模型输出。

后续修改可以在新边界逐项推进，但修改prompt、模型输入、校验启用位置或恢复策略都应另立实验版本。旧文件基线哈希不是不可变引擎镜像；有意修改旧引擎后，需要明确记录变更，不能悄悄重写基线冒充兼容。
