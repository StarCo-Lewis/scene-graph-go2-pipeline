# GO2 端补丁与服务管理说明（2026-10-30）

本包提供今日整理的**当前源码补丁集**，不是基于已确认狗端 commit 的差分补丁，也未读取狗上安装版本。先比对、备份，再安装。已有的导航组程序、地图、ROS2 环境继续保留。

## 内容与作用

| 文件 | 用途 |
|---|---|
| `go/tcp_bridge.py`、`go/tcp_protocol.py` | 我们的 TCP 8888 → ROS2 `/nav/navigate_to` 网桥，包含请求标识、反馈/终态/路径、日志与未完成任务保护 |
| `embodied_pipeline/deployment/configure_first_alignment_only.py` | 将已识别的 `mission/ros_action.py` 中 `realign_on_mismatch` 改为 False；首段仍建立对齐，后续不能自动重设原点 |
| `embodied_pipeline/deployment/configure_fine_stop_4cm.py` | 位置精调停止半径改为 0.04 m，支持已识别公式及 `cfg.stop_radius_m` 版本；默认只显示 diff |
| `dog_services.py`、`manage_dog_services.ps1`、根目录 CMD | Windows 点击启动/重启狗端服务，密码手动输入终端 |
| `dog_sync_time.sh` | 在电脑 Git Bash 中运行的对时脚本，不能在狗上执行 Windows 命令 |
| `ros2_client_ws/src/nav_interfaces` | Action 定义快照，用于接口核对，不能代替导航服务器 |

不包含可调成功容差补丁：此前该功能已撤回。4 cm 是位置精调停止条件，现有 goal 的 pos_tol 仍为 5 cm。默认请求 timeout=120 s、max_speed=0.5 m/s 等客户端参数不在此补丁中修改；max_speed 字段不是实测限速保证。

## 前提与检查

狗上需要已有 `~/nav_ws/start_capo.sh`、`start_nav_real.sh`、`dogenv_run.sh`、导航仓库 `~/nav_ws/nav`、地图以及正确的 ROS2 Foxy / nav_interfaces 环境。已有网桥目录为 `~/nav_ws/nebula_bridge`。

执行安装前，停止客户端自动队列，确认狗已停止，并按导航组手册停止正在运行的导航节点和网桥。不要在运动中替换文件或重启。不要删除 `runtime/pending_goal.json` 来强行解除未完成状态；先确认任务终态并按原手册归档。服务重启会使既有导航对齐失效。

## 上传（Windows PowerShell，进入解压根目录）

```powershell
scp .\go	cp_bridge.py .\go	cp_protocol.py unitree@192.168.123.18:~/
scp .\embodied_pipeline\deployment\configure_first_alignment_only.py .\embodied_pipeline\deployment\configure_fine_stop_4cm.py .\embodied_pipeline\deployment\dog_services.py unitree@192.168.123.18:~/
```

SSH 密码按提示输入。这里先上传到 home，未覆盖运行目录。

## 狗端备份与补丁预览（狗上的 Bash）

以下命令只在服务已停止后执行；保留现有 .token 和 runtime，不从压缩包替换它们。

```bash
backup_dir="$HOME/nav_ws/nebula_bridge/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"
cp -p ~/nav_ws/nebula_bridge/tcp_bridge.py ~/nav_ws/nebula_bridge/tcp_protocol.py "$backup_dir/"
cp ~/tcp_bridge.py ~/tcp_protocol.py ~/nav_ws/nebula_bridge/
python3 ~/configure_first_alignment_only.py ~/nav_ws/nav/mission/ros_action.py
python3 ~/configure_fine_stop_4cm.py ~/nav_ws/nav/control/fine_approach.py
```

审阅 diff 后，再运行对应的 `--apply`。两种源码补丁均会保存带时间的 `.bak`；结构不识别时停止，不应手工绕过报错。

```bash
python3 ~/configure_first_alignment_only.py ~/nav_ws/nav/mission/ros_action.py --apply
python3 ~/configure_fine_stop_4cm.py ~/nav_ws/nav/control/fine_approach.py --apply
```

4 cm 补丁不会自动部署其它控制器改动，不修改 CAPO 尺度、速度、yaw 容差或最终 success 判定。

## 启动 / 重启

在 Windows 双击 `启动或重启狗端服务.cmd`。选 1 启动全部，2 只重启导航+网桥，3 全部重启，4 查看状态。密码直接输入终端；确认狗和客户端队列已停止后输入 `STOPPED`。

该脚本只上传管理脚本，不自动安装 bridge 或应用 4 cm 补丁，所以必须先完成上面的部署。脚本启动导航组既有程序，保留其当前现场配置：CAPO 尺度 1.048、619 5 cm 栅格、waypoint 模式；仅适用于该现场约定。没有自动发送导航目标。

若从狗端操作：

```bash
python3 ~/dog_services.py status
python3 ~/dog_services.py restart-nav
```

首次启动需现场口令文件 `~/nav_ws/nebula_bridge/.token` 已存在，至少 24 字符；本包不含口令。共享口令在客户端手动填写。ROS_DOMAIN_ID 使用现场的 0，scene-frame 为 `d3_619_slam_world_x_forward_y_left_z_up_m`；改名称不等于物理配准。

启动成功后，客户端重新握手，确认 Action 可用、motion_enabled=True；测量当前场景坐标和朝向，核对目标后再由操作员发送。日志在 `~/nav_ws/pipeline_service_logs/`，bridge 日志在 runtime。失败不会自动重试目标。

## 回滚与问题处理

停止服务后，把备份 bridge 两文件恢复到 nebula_bridge；把本次补丁生成的 `.bak` 恢复到对应源文件，然后按同一流程重启并重新核对起点。不要恢复或清除任务状态文件来伪造空闲。

8888 被占用：先查看 `ss -ltnp 'sport = :8888'`，核实进程身份；不要重复启动多个网桥。没有监听但仍绑定失败时等待旧连接释放。环境报 `nav_interfaces` 不存在：使用导航组的 dogenv_run.sh 加载已编译工作区，不要把 Windows 二进制搬过去。

本包只做离线语法/协议检查；具体狗端补丁是否已应用、当前控制器状态及真实到达精度，需要现场验证。
