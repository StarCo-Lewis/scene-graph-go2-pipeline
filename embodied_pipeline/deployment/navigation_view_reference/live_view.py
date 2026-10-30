# -*- coding: utf-8 -*-
"""
tools/live_view.py —— 笔记本端的【实时可视化】+ 数据落盘

画什么（2D 俯视图）:
    ① 场景图栅格（组1 的图，本地文件）
    ② 规划路径  ← /nav/planned_path（任务开始时发一次）
    ③ 狗的位置 + 朝向  ← /nav/feedback（10Hz）
    ④ 走过的轨迹  ← 本地累积 feedback 里的位姿

★ 为什么不传图片：
   图片几万~几十万字节 —— 正是那条【传不过来】的通道。
   feedback 一条才 80 字节，笔记本本地渲染 ⇒ 差 100 倍。

★★ 2026-09-23 补：除了存图片，还把【轨迹数据】落盘成 npz。
   为什么：2026-09-23 真机首跑时，现场观测和日志数字对不上，
   但只有图片、没有数据，事后没法复核。npz 里有：
     traj/traj_t      轨迹 (x,y,yaw) + 每条的时间戳
     stages           每个 feedback 的 (stage, remaining_dist) 时序
     path_0..N        每条任务下发的规划路径（保持原样，未插值）
     n_path/n_result  条数

用法（笔记本，需要 ROS2 环境）:
    source ~/capo_live/env.sh
    python3 tools/live_view.py --grid demo_scene.npz
    # 默认一直跑到 Ctrl+C（Ctrl+C 会落盘）

离线复看（不连狗，直接画已存的 npz）:
    python3 tools/replay_view.py tools/live_view.npz --grid demo_scene.npz
"""
import argparse
import json
import signal
import sys
import threading
import time
import os

import numpy as np
import matplotlib
matplotlib.use('TkAgg' if os.environ.get('DISPLAY') else 'Agg')
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

ap = argparse.ArgumentParser()
ap.add_argument('--grid', default='/mnt/d/迷雾通/nav/demo_scene.npz')
ap.add_argument('--scene-origin', nargs=2, type=float, default=None,
                help='组1 场景系原点在图自己坐标里的位置；不给就读 npz 里存的')
ap.add_argument('--out', default='/mnt/d/迷雾通/nav/tools/live_view.png',
                help='实时快照的图片路径')
ap.add_argument('--npz', default='/mnt/d/迷雾通/nav/tools/live_view.npz',
                help='★ 轨迹数据落盘（不只存图片）—— 跑完还能再画、再分析')
ap.add_argument('--save-every', type=float, default=5.0,
                help='每隔几秒存一张图（无显示器时用）')
ap.add_argument('--seconds', type=float, default=1e9,
                help='跑多久自动退出；默认极大 = 一直跑到 Ctrl+C')
a, _ = ap.parse_known_args()


class Viewer(Node):
    def __init__(self):
        super().__init__('nav_live_view')
        qr = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                        history=HistoryPolicy.KEEP_LAST)
        self.pose = None            # (x, y, yaw)
        self.traj = []
        self.traj_t = []            # ★ 对应的时间戳
        self.path = None            # 最后一条
        self.paths = []             # ★ 所有任务的路径
        self.results = []           # ★ 所有 result 的原始 dict
        self.stages = []            # ★ (t, stage, remaining) 时序
        self.n_result = 0
        self.stage = '-'
        self.remain = float('nan')
        self.lock = threading.Lock()
        self.create_subscription(String, '/nav/feedback', self._fb, qr)
        self.create_subscription(String, '/nav/planned_path', self._path, qr)
        self.create_subscription(String, '/nav/result', self._res, qr)

    def _fb(self, m):
        try:
            d = json.loads(m.data)
            t = time.time()
            with self.lock:
                self.pose = (float(d['cur_x']), float(d['cur_y']),
                             float(d['cur_yaw']))
                self.traj.append(self.pose)
                self.traj_t.append(t)
                self.stages.append((t, d.get('stage', '-'),
                                    float(d.get('remaining_dist', float('nan')))))
                self.stage = d.get('stage', '-')
                self.remain = float(d.get('remaining_dist', float('nan')))
        except Exception:
            pass

    def _path(self, m):
        try:
            d = json.loads(m.data)
            with self.lock:
                self.paths.append(d['waypoints'])   # ★ 每个任务的路径都留着
                self.path = d['waypoints']
        except Exception:
            pass

    def _res(self, m):
        try:
            d = json.loads(m.data)
            with self.lock:
                self.results.append(d)
                self.n_result += 1
            print('  [result] %s' % d, flush=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def snapshot(self):
        with self.lock:
            return (self.pose, list(self.traj), list(self.paths),
                    self.stage, self.remain)

    def save_data(self, path):
        """★ 把轨迹/路径/结果落盘 —— 只存图片的话跑完就没法再分析了。"""
        try:
            with self.lock:
                traj = np.array(self.traj) if self.traj else np.zeros((0, 3))
                tt = np.array(self.traj_t) if self.traj_t else np.zeros((0,))
                st = list(self.stages)
                paths = [np.array(p) for p in self.paths]
                results = list(self.results)
            kw = dict(traj=traj, traj_t=tt,
                      n_path=len(paths), n_result=len(results))
            for i, p in enumerate(paths):
                kw['path_%d' % i] = p
            if st:
                kw['stage_names'] = np.array([s[1] for s in st])
                kw['stage_t'] = np.array([s[0] for s in st])
                kw['stage_remain'] = np.array([s[2] for s in st])
            np.savez(path, **kw)
            print('  ★ 数据已落盘: %s  （轨迹 %d 点，路径 %d 条，result %d 条）'
                  % (path, len(traj), len(paths), len(results)), flush=True)
            for r in results:
                print('    result: success=%s reason=%s elapsed=%s'
                      % (r.get('success'), r.get('reason'), r.get('elapsed')),
                      flush=True)
        except Exception as e:
            print('  落盘失败: %s' % e, flush=True)


def draw(ax, occ, origin, res, traj, paths, pose, stage, remain, title_extra=''):
    H, W = occ.shape
    ax.clear()
    ax.imshow(occ, origin='lower', cmap='Greys', vmin=0, vmax=1.35,
              extent=[origin[0], origin[0] + W * res,
                      origin[1], origin[1] + H * res], zorder=1)
    for i, wp in enumerate(paths or []):
        p = np.array(wp)
        if len(p) < 2:
            continue
        ax.plot(p[:, 0], p[:, 1], color='tab:red', lw=1.5, ls='--', alpha=0.7,
                zorder=3, label='planned path' if i == 0 else None)
    if traj:
        t = np.array(traj)
        ax.plot(t[:, 0], t[:, 1], color='tab:blue', lw=2.2, zorder=4,
                label='travelled')
        ax.plot(t[0, 0], t[0, 1], 's', color='k', ms=8, zorder=5,
                label='start')
    if pose:
        x, y, yaw = pose
        ax.plot(x, y, 'o', color='lime', ms=12, mec='k', zorder=6)
        ax.annotate('', xy=(x + 0.6 * np.cos(yaw), y + 0.6 * np.sin(yaw)),
                    xytext=(x, y),
                    arrowprops=dict(arrowstyle='-|>', color='k', lw=2.2),
                    zorder=7)
    ax.set_title('stage: %s     remaining: %.2f m%s'
                 % (stage, remain, title_extra), fontsize=12)
    ax.grid(alpha=0.25)
    ax.set_aspect('equal')
    ax.set_xlabel('scene x [m]')
    ax.set_ylabel('scene y [m]')
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, loc='lower right')


def main():
    d = np.load(a.grid)
    occ = d['occ'].astype(bool)
    mo = d['origin']
    res = float(d['res'])
    # ★ 默认从 npz 里读 scene_origin（别在代码里写死 —— 会画偏）
    if a.scene_origin is not None:
        so = np.array(a.scene_origin)
    elif 'scene_origin' in d:
        so = np.array(d['scene_origin'])
    else:
        so = np.zeros(2)
    origin = (float(mo[0] - so[0]), float(mo[1] - so[1]))
    print('场景系原点在图里的位置: %s   图在场景系里的 origin: (%.2f, %.2f)'
          % (so, origin[0], origin[1]), flush=True)

    # ★ SIGTERM 也要落盘 —— `pkill` 发的是 SIGTERM，不是 Ctrl+C 的 SIGINT。
    #   不接的话：脚本被 pkill 掉 ⇒ finally 不执行 ⇒ npz 没写 ⇒ 白跑。
    def _term(sig, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)

    rclpy.init()
    v = Viewer()
    threading.Thread(target=rclpy.spin, args=(v,), daemon=True).start()
    print('等 /nav/feedback ...（Ctrl+C 退出）', flush=True)

    fig, ax = plt.subplots(figsize=(10, 8.5))
    t0 = time.time()
    t_last_save = 0.0
    try:
        while time.time() - t0 < a.seconds:
            time.sleep(0.2)
            pose, traj, paths, stage, remain = v.snapshot()
            paths = [paths[-1]] if paths else []      # 实时只画最新那条
            ax.clear()
            draw(ax, occ, origin, res, traj, paths, pose, stage, remain)
            plt.tight_layout()
            plt.pause(0.01)
            if time.time() - t0 - t_last_save > a.save_every:
                t_last_save = time.time() - t0
                try:
                    plt.savefig(a.out, dpi=100)
                    print('  已存快照 %s  (stage=%s, remain=%.2f)'
                          % (a.out, stage, remain), flush=True)
                except Exception as e:
                    print('  存图失败: %s' % e, flush=True)
                # ★★ npz 也【周期性】存 —— 不能只靠退出时的那一次。
                #    2026-09-23 踩过：`pkill` 发 SIGTERM，而主线程正阻塞在
                #    plt.pause()（C 代码）里，信号处理器没机会跑 ⇒ finally 不执行
                #    ⇒ 图片存了、数据没存。周期存就不怕被硬杀。
                v.save_data(a.npz)
    except KeyboardInterrupt:
        print('\n  Ctrl+C -> 收尾', flush=True)
    finally:
        try:
            pose, traj, paths, stage, remain = v.snapshot()
            ax.clear()
            draw(ax, occ, origin, res, traj, paths, pose, stage, remain,
                 '   [FINAL]')
            plt.tight_layout()
            plt.savefig(a.out, dpi=100)
            print('最后一张图: %s' % a.out, flush=True)
        except Exception as e:
            print('存终图失败: %s' % e, flush=True)
        # ★ 落盘放在最前，且不用 os._exit 之前的任何提前退出
        v.save_data(a.npz)
        sys.stdout.flush()
        os._exit(0)


if __name__ == '__main__':
    main()
