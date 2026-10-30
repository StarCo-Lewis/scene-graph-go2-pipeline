# 输入场景图到工程 JSON：更新与使用（2026-10-30）

这是完整依赖覆盖式补丁：`payload/` 内保持原仓库相对路径，含当前 embodied_pipeline 和所需 outputs 旧引擎源码。它不是一个独立 wheel，也不是只复制一个 py 文件的修复。建议先解压到新目录检查；覆盖原项目时先备份清单中同名文件，关闭客户端/规划进程，勿覆盖正在运行的批次。原始运行结果不在补丁中，不能删除。

在 GitHub 全工程包中，下述文件直接位于仓库根目录，没有 payload 层。

## 本次改动

- 取消当前规划入口的 cosine/top-k 技能预筛选，使用 `full_inventory_no_cosine_v1`。每个子任务获得当前入口加载技能库的完整候选，参数与依赖校验仍执行；不再启动本地 embedding。
- 多目标的中间访问是阶段里程碑，最终状态只表达最后目标；A→B→A 保留全部三段。
- 客户端区分单次任务与多目标队列，自动生成候选导航点并保留真实终态/轨迹/任务日志；不会因为规划成功自动运动。
- BBOX 周边采样后按真实占据图检查终点与圆形净空，避免把目标落入障碍物。仅用于导航映射阶段，不改变上游工程 JSON，也不代替导航组的路径规划。

## 输入契约

真正需要的是**图 + edges + 元数据 + 指令**，只有两个几何文件不足以可靠声明坐标系。

- 图：JSON 数组。每个对象有唯一整数 `id`、`object_tag`、`caption`、`bbox_center=[x,y,z]`、`bbox_extent=[完整尺寸x,y,z]`。
- edges：当前引擎支持的关系 JSON 数组；不能为缺失关系虚构 on/in。合法空数组可表示示例中未提供关系，不等于证明场景无关系。
- 元数据：当前实景入口检查 `schema_version=real_scene_import_v1`、准确的 node_count/edge_count，以及 `coordinate_frame=d3_619_slam_world_x_forward_y_left_z_up_m`；客户端还要求 `unit=metre`、`up_axis=z`。这一入口是 D3-619 坐标配置，不是任意地图的通用转换器。
- 指令：英文自然指令；多目标使用 ordered_goals。物体 ID 可显式写入，也可由语义绑定确定；保留不确定/拒绝，不虚构结果。

`prepare_graph_task.py` 接受任意输入文件名，例如 scene_graphs.json 和 edges.json，复制成内部规定的 scene_graph.json / scene_graph_edges.json，并计算哈希。**不会转换坐标，也不会把错误坐标通过改名“修正”。** 元数据必须由场景提供方确认。

## 准备与离线检查

在 payload 或完整工程根目录运行：

```bash
python tools/prepare_graph_task.py --graph /path/to/scene_graphs.json --edges /path/to/edges.json --metadata /path/to/scene_metadata.json --instruction "Navigate to table ID 179, then navigate to sofa ID 189." --navigation-mode ordered_goals --output outputs/my_task_01
python -m embodied_pipeline real-scene-plan --task-json outputs/my_task_01/task.json --output-dir outputs/my_preflight_01 --preflight-only
```

需要在实际规划使用的操作系统中准备 task.json：不要把 Windows 绝对路径直接给 WSL/Linux。可以先使用 examples/synthetic_scene 的合成数据验证文件格式，不能用于真机。

## 调用 API（仅在自行确认额度后运行）

Linux / WSL Bash：

```bash
read -rsp 'DashScope API key: ' DASHSCOPE_API_KEY
printf '\n'
export DASHSCOPE_API_KEY
python -m embodied_pipeline real-scene-plan --task-json outputs/my_task_01/task.json --output-dir outputs/my_plan_01 --model qwen3.8-flash --base-url https://dashscope.aliyuncs.com/compatible-mode/v1
unset DASHSCOPE_API_KEY
```

模型名沿用现场配置，不保证你的账户有权限；可指定经确认可用的兼容模型。不自动重跑；新运行使用新目录。Windows 可用标准库隐藏输入后设置环境，或者在客户端 API 页填密钥。本说明不要求 embedding、torch 或本地模型下载。

## 交付文件与状态

规划输出包括 task_input、输入图副本、object_alignment、skill_retrieval、action_codebook、llm_matrix_proposal、decoded_matrix、skill_plan、static_review、engineering_plan.json 和 planning_summary.json；实际是否生成取决于失败阶段，不能伪造缺失产物。

- `planning_json_generated_not_executed`：静态检查通过，尚未执行。
- `planning_output_requires_review`：输出需要检查，可能有 JSON 但不能视为通过。
- `planning_failed_not_executed`：调用或流程失败，查看 summary 和请求原日志。

当前 real-scene-plan 是导航测试配置；其校验不承诺非导航操作有效。需要扩展到完整操作任务时须单独声明技能与物理初态，不能借本补丁声称抓取/放置已经实测。

## 版本与兼容

补丁附逐文件 SHA256，使用 `python tools/verify_release.py` 核对。历史 legacy_manifest 保留原样，已知旧基线有改动；它不是本补丁验收清单。冻结的采集批次有代码哈希检查，不应把旧结果当作新策略续跑；新实验另建版本。回滚时恢复原文件备份，不删除实验输出。

本次未发起 API 请求、未访问 WSL、未向机器人发送命令。修改后的输入 token 可能增加；评估效果需要后续明确授权的新实验。


## 换电脑必做：客户端路径依赖配置

**当前客户端仍保留作者机器的默认路径，不能解压后直接沿用。界面中的运行配置也不应假定会跨重启保存。** 按下表配置后再调用 API；无需为此启动狗或发送导航。

| 位置 | 需要检查/修改 | 如何填写 |
|---|---|---|
| 启动客户端的 Python | Windows 启动脚本优先使用 FIELD_UI_PYTHON，否则会尝试作者 Codex 运行时再回退 python | 建议用你自己的 Python 显式启动；必须能 import tkinter |
| 默认场景目录 | field_backend.py 的 DEFAULT_SCENE 指向仓库下原 D3-619 输出目录；源码包未包含该目录 | 用启动参数 --scene 指向新电脑的真实场景目录，或启动成功后用“加载场景”选择目录 |
| “连接 / 规划环境 / 反馈” → “WSL 规划 Python” | 默认 /home/lewis/miniforge3/envs/conceptgraph/bin/python | 改为该电脑 WSL 中实际用于规划的 Python 完整路径，不能填写 Windows python.exe |
| “API 规划” → “WSL 发行版” | 空白表示系统默认发行版，可能不是安装环境的那一个 | 在 Windows 用 wsl -l -v 确认名称，必要时填写准确名称 |
| 导航占据图 | navigation_plot.py 的 DEFAULT_GRID 为包内 ui/assets/navigation_619/619_map_v1_clean_5cm.npz | 把已核对的私有地图复制到这个相对位置；如另存位置需修改 DEFAULT_GRID。终点筛选与绘图应使用同一地图 |
| 已有工程计划 / 轨迹日志 | 原 task.json、summary 或 manifest 可能含旧电脑绝对路径 | 重新选择新位置；新任务用 prepare_graph_task.py 重建哈希与路径，不把旧绝对路径直接交给新系统 |
| GO 管理脚本 | manage_dog_services.ps1 的 dogAddress；dog_services.py 的 WS/BRIDGE 和导航地图路径 | 默认 unitree@192.168.123.18、~/nav_ws；狗端布局不同时同步修改这些设置与 README 命令 |
| ROS2 直连 | 启动客户端进程需要继承本机 ROS 和 nav_interfaces 的环境 | 在已 source 正确 ROS/工作区的同一终端启动；Windows Python 不能直接 import WSL 的 rclpy |

### 推荐的 Windows 启动方式

在 PowerShell 中进入解压后的工程根目录。将示例路径替换为本机路径：

```powershell
cd "D:\Projects\go_pipeline"
python -c "import sys, tkinter; print(sys.executable)"
python -m embodied_pipeline field-console --scene "D:\PrivateData\D3619\scene"
```

若使用双击 CMD 启动，可先设置 FIELD_UI_PYTHON，或修改 CMD 中解释器选择；只有在同一终端中设置环境变量后启动 CMD，该次进程才会继承它。更稳妥的是首次使用上述显式命令。不要复制作者 C:/Users/1 或 /home/lewis 路径当作自己机器的路径。

### 查找 WSL 规划解释器

先在 Windows 执行 `wsl -l -v`。然后由你在目标 WSL 发行版中激活准备好的规划环境，执行：

```bash
python -c "import sys; print(sys.executable)"
```

将输出完整路径填入客户端“WSL 规划 Python”。例如另一位同学可能是 /home/runtangtang/miniconda3/envs/ros2/bin/python3，但必须以实际输出和该环境能否导入本工程为准，不要求环境名称与作者一致。当前图→JSON 不再需要 embedding Python 或 embedding 模型目录，代码里的旧兼容字段不会用于 cosine 筛选。

### Windows 与 WSL 的文件路径不能混用

- Windows 文件选择器：D:\Projects\go_pipeline。
- 同一目录在默认 WSL 挂载下：/mnt/d/Projects/go_pipeline。
- Git Bash 的 /d/Projects/go_pipeline 不是 WSL 路径。
- GUI 规划入口会将 Windows 驱动器路径转成 /mnt/<盘符>/...，所以工程和输入应在 WSL 可访问的普通本地盘目录。网络共享路径、\\wsl$ 路径及自定义挂载不属于当前自动转换支持范围。
- 在 Linux/WSL 手工运行 prepare_graph_task.py 和规划时，使用 Linux 路径，并在该系统中生成 task.json。无需修改代码里的 ROOT：它从代码位置计算。

迁移完成先执行离线 preflight-only，再启动 GUI 检查场景能加载、地图能显示、WSL Python 路径正确；之后才按需调用 API。连接狗时还需核对 IP、共享口令与真实坐标配准，这些不是路径修改能够代替的。
