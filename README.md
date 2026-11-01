# 场景图任务规划与 GO2 导航联调工程

源码交付版本：**2026-11-01-u87-semantics-source-v1**。这是维护者当前源码的独立交付快照，尚未由本包创建 git commit、推送 GitHub 或部署现场。此包未包含导航组 CAPO/控制器的完整实现、ROS 安装、模型权重、真实场景图或现场日志。未在另一台机器人上重新验证部署。

本次更新：保留 U87 的 graph、edges、metadata 三 SHA 绑定，并保留当前语义审计扩展；新增终态提示 v2、批量采集的逐条 JSON 留档与 usable_as_plan 独立判定。最新 12 条外部 qwen3.8-flash 小批已有实际结果：8 条完整符号计划符合该批参考，4 条合理停止且保存不完整 JSON；该小样不代表通用正确率或真机通过。详见 [本次版本及RC对照](docs/版本更新_20261101.md) 和 [metadata 补丁与符号小批验证](docs/metadata补丁与符号小批验证.md)。后者是历史说明，当前进度以本次版本文件为准。

## 已实现的完整链路

场景图 + edges + 坐标元数据 + 英文任务 → LLM 分解 → 物体绑定 → 完整技能候选 → 技能矩阵 → 静态校验 → engineering_plan.json → 客户端导航点映射 → TCP 8888 网桥或 ROS2 Action → 导航组 `/nav/navigate_to` → GO2 反馈 → 界面和日志。

- 当前技能候选策略为 `full_inventory_no_cosine_v1`，不做 cosine/top-k 筛选，也不启动 embedding。完整指**当前入口加载的技能库**；实景导航入口仍只加载既有五个符号技能，不能据此声称所有教师技能均已接入硬件。
- 实景入口是导航专用校验配置。抓取、放置等模块保留在源码中，不等于狗端已经支持；不是通用操作任务的真机执行接口。
- 系统 1：到 A 后再次输入 B；沿用有效的实际终态位姿。系统 2：一次输入 A then B，按顺序执行，成功到达后停留 10 秒再发下一段。
- 导航服务首段建立场景对齐，后续保持同一坐标系；服务重启后重新测量，不能沿用旧原点。
- BBOX 外侧距离默认 0.4 m，窗口可调。环周候选点须通过 619 静态占据图和半径 0.35 m 圆形净空检查，无点则拒绝。此为终点筛选，不是本方路径规划器，不证明动态避障或跨图配准成立。
- GO 结果决定成功/失败；规划通过不等于执行成功。保留目标、实际轨迹、反馈和日志。
- 末端 4 cm 补丁只修改位置精调停止条件；请求 `pos_tol=0.05` 未随之更改。两个指标不同，不承诺最终误差一定小于 4 cm。

## 环境与依赖

| 部署位置 | 已知环境 / 要求 |
|---|---|
| Windows 客户端 | 现场曾用 Python 3.12 + Tk；Windows 11；网线连接 GO2 |
| 作者规划环境 | Ubuntu 24.04 / WSL 2.7.12.0，Linux 6.18.33.2-2；历史规划 Python 3.10；这些是先前记录，未重新探测 |
| GO2 | 现场日志为 Ubuntu、Python 3.8、ROS2 Foxy；需已有 CAPO、导航节点、`nav_interfaces` |
| 其他电脑 | Ubuntu 22.04 / ROS2 Humble 的接口需在本机构建并 source，不能直接拷贝 Foxy 编译产物 |

纯图→JSON 的当前路径使用标准库；无需 torch / sentence-transformers。Tk 窗口需要 Python 安装包含 tkinter。ROS2 直连另需 rclpy 和已构建的 nav_interfaces；TCP 客户端不需要本机 ROS2。原始建图/仿真/历史实验有额外依赖，不属于本次输入图已准备好后的运行前提。Python 环境和 ROS 环境不要直接互相覆盖。

## 快速开始（在解压后的仓库根目录）

```bash
python -m embodied_pipeline inspect
python tools/verify_release.py
python tools/prepare_graph_task.py --graph examples/synthetic_scene/scene_graph.json --edges examples/synthetic_scene/scene_graph_edges.json --metadata examples/synthetic_scene/scene_metadata.json --instruction "Navigate to table ID 1, then navigate to chair ID 2." --navigation-mode ordered_goals --output outputs/local_task_01
python -m embodied_pipeline real-scene-plan --task-json outputs/local_task_01/task.json --output-dir outputs/local_preflight_01 --preflight-only
```

以上例子明确为**合成接口演示**，不是 D3-619 地图，不能用于真机。完整规划命令和返回状态见 [图到工程JSON说明](docs/图到工程JSON说明.md)。每次运行使用新输出目录。

```bash
python -m embodied_pipeline field-console --scene /absolute/path/to/your/scene
```

Windows 可以双击 `启动D3619现场窗口.cmd`，但需要先把私有场景文件放到原默认目录，或使用上面的 `--scene` 显式传入。GUI 的 API 按钮仍通过配置的 WSL Python 运行规划，须填写本机实际路径；Linux 同学可先用 CLI 生成 JSON，再在窗口加载。不要假定 Windows/WSL/ROS2 运行环境可直接互换。

## 现场地图与接口

真实地图未放入 GitHub 包。现场使用时将已核对版本的 `619_map_v1_clean_5cm.npz` 放到 `embodied_pipeline/ui/assets/navigation_619/`；其已知 SHA256 为 `bc56e8d1e48fd363101723d10c5e392d22b40f65c8729fe88b11a0c8bf37effd`。不要以其他地图替换后仅修改 frame 名称。语义图、占据图和狗定位的实际配准需要现场确认。

狗端已有地图路径为 `~/nav_ws/nav/tools/619_map_v1_clean_5cm.npz`。狗端部署及补丁顺序见 [GO补丁说明](docs/GO补丁说明.md)。默认网络为 `192.168.123.18:8888`，协议 `nebula.navigation.tcp.v1`，`ROS_DOMAIN_ID=0`。电脑通过 USB 网卡接网线，TCP 是网线上的应用通信协议，两者并不互斥。

Action 定义：`embodied_pipeline/deployment/ros2_client_ws/src/nav_interfaces/action/NavigateTo.action`。输入 `s_*`、`g_*` 六维位姿及 `pos_tol/yaw_tol/timeout/max_speed`；当前主要用 x/y/yaw，其他值依合同填入。坐标为米、弧度，x 向前、y 向左、z 向上。以本文件和服务器相同版本为准。

## 配置与日志

- `DASHSCOPE_API_KEY`：仅规划进程环境或客户端内存输入；不写入仓库。
- `FIELD_BRIDGE_TOKEN`：狗端 `.token` 加载的共享口令；客户端连接时手动输入，不能使用 API key 代替。
- `ROS_DOMAIN_ID=0`：现场当前域；`RMW_IMPLEMENTATION` 按导航组环境设置，本包不擅自修改。
- `FIELD_UI_PYTHON`：可选，Windows 客户端解释器路径。
- 规划：各运行目录中的 input/raw/codebook/review/engineering_plan 和 summary；失败时保留原响应，不伪造结果。
- 客户端导航：`outputs/real_scene_runs/tcp_navigation_*`、`navigation_view_*`；轨迹图/索引以 manifest 为准。
- 狗端网桥：`~/nav_ws/nebula_bridge/runtime/bridge_events.jsonl`；服务管理日志：`~/nav_ws/pipeline_service_logs/`。

## GitHub 上传

若更新已有仓库：将本包根目录中的源码、docs、README、PACKAGE_MANIFEST 等复制到已有仓库目录，保留原 `.git`，然后检查差异再提交。不要把本包顶层文件夹作为额外一层目录提交，也不要上传 ZIP 本身来替代源码更新。详见 [GitHub更新步骤](docs/GitHub更新步骤_20261101.md)。首次建立空仓库才使用下面的 git init 示例。

```bash
git init
git add .
git status --short
git commit -m "Import graph planning and GO2 navigation integration"
# 然后按你的 GitHub 空仓库提示设置 remote 并 push。
```

`.gitignore` 排除了运行数据、口令、模型及私有地图；`outputs/` 下被明确列入的旧引擎源码和技能配置是依赖，不能全部删除。不要在此包内运行旧实验启动脚本来复现本次流程。部分历史脚本保留作者路径及旧 cosine 设计，当前入口已停用这些筛选。

`legacy_manifest.json` 是早期基线，不能代表当前发布版本。当前完整性以 `PACKAGE_MANIFEST.json` 和 ZIP SHA256 为准；旧 verify-legacy/Oracle 完整性检查可能失败，不要覆盖旧基线伪装通过。依赖历史私有数据的测试无法在此源码包全量运行。

本包未自动授予开源许可证。第三方接口/参考代码的原注释保留；公开前由项目负责人确认代码和资源的发布权限。未包含真实密钥、原始实验批次、机器人运行日志、学校场景地图及点云。这里的源码测试和打包验证均不构成真机实测结论。


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
