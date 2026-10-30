# 共享接口边界

这里声明模块职责和可替换调用签名，不另造一套模型JSON。所有运行期字段继续采用旧引擎的数据结构。

| 边界 | 输入 | 输出/副作用 |
| --- | --- | --- |
| 感知/环境 | 场景标识或原始节点/边 | 原cfg、nodes、edges、来源hash；不新增房间连接或物理验证 |
| 规划 | 原始英文指令、完整契约、当前状态、场景快照 | 原alignment：对象角色、子任务、依赖、初始断言、目标及对象绑定 |
| 指令技能 | alignment、技能库、当前码表、余弦候选 | 原四列矩阵和解码后的plan；不宣称执行成功 |
| 校验 | 原始输出、码表、alignment或状态 | 原函数返回值或原异常；不在重组中统一改成新的status外壳 |
| 执行 | plan、alignment、技能库、节点、状态、compact函数 | 原execution_feedback：trace、计数、failure、resulting_state；失败前缀保留 |
| 异常处理 | 原异常或无提案状态 | 保持原分类、拒绝与停止边界，不自动生成替代动作 |
| 记忆 | descriptor、开始状态、history、execution_feedback、cfg与nodes | 原state/history事务及日志；模型goal不能赋值给memory |
| 调度 | args、episode、output folder、encoder | 原episode/batch summary及相同工件；不插入额外推理阶段 |

`EpisodeServices`允许测试时显式注入proposer、executor或scene_loader，不提供任何Oracle答案，也不默认启用替换。离线测试使用的参考动作只在tests中导入，生产模块不得导入测试参考。

未来Oracle-D和Oracle-B必须遵守已有角色/码表/输入快照契约，单独记录干预版本及来源。当前分解+绑定的旧联合子进程没有被拆开；不能仅因这里存在接口就声称已经实现独立Oracle实验。

`ExecutionBackend`只定义调用形状。`RobotBackend.execute`当前始终抛出NotImplementedError，不会模拟一个成功的真机回包。

## 控制器分发框架（2026-10-30）

新增可选的离线入口 `python -m embodied_pipeline dispatch-preview`，读取现有工程JSON，保留动作与参数，将技能映射到其他组的逻辑控制模块。具体字段、适配器方法、反馈状态和待对接事项见 [Controller dispatch framework](../execution/CONTROLLER_DISPATCH.md)。此入口不连接真机；`--simulate`仅验证合成反馈下的串行流程，不替换原有执行器或现场窗口。
