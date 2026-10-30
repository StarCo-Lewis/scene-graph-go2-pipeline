# -*- coding: utf-8 -*-
"""
tools/replay_view.py —— 离线复看：把 live_view.py 存下来的 npz 画出来

为什么要它
----------
2026-09-23 真机首跑时，**现场观测和日志数字对不上**（用户看到狗在主动转向，
但日志只有每 1 秒一行的状态）。当时没有数据可查、只能靠图片看。
⇒ 现在 live_view.py 会把轨迹落盘，这个脚本负责**事后**把它画出来：
   轨迹 / 规划路径 / 起点终点 / 阶段时序，不连狗、不连 ROS。

用法（笔记本 WSL）:
    python3 tools/replay_view.py tools/live_view.npz --grid demo_scene.npz
    python3 tools/replay_view.py xxx.npz --out my.png
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz', help='live_view.py 存下来的 npz')
    ap.add_argument('--grid', default=None, help='场景图 npz（不给就只画轨迹）')
    ap.add_argument('--out', default=None, help='输出 png；默认同名 .png')
    ap.add_argument('--curve', action='store_true',
                    help='额外画 距离-时间 / 航向-时间 两张曲线')
    a = ap.parse_args()

    out = a.out or (os.path.splitext(a.npz)[0] + '_replay.png')
    d = np.load(a.npz, allow_pickle=True)

    traj = np.array(d['traj']) if 'traj' in d.files else np.zeros((0, 3))
    tt = np.array(d['traj_t']) if 'traj_t' in d.files else np.zeros((0,))
    n_path = int(d['n_path']) if 'n_path' in d.files else 0
    n_res = int(d['n_result']) if 'n_result' in d.files else 0
    paths = [np.array(d['path_%d' % i]) for i in range(n_path)
             if ('path_%d' % i) in d.files]

    print('=' * 60)
    print('  离线复看: %s' % a.npz)
    print('=' * 60)
    print('  轨迹 %d 点 ｜ 规划路径 %d 条 ｜ result %d 条' % (len(traj), n_path, n_res))

    if len(traj) == 0:
        print('  ✗ 没有轨迹数据（feedback 一条都没收到？）')
        sys.exit(1)

    if len(tt) > 1:
        print('  记录时长 %.1f s' % (tt[-1] - tt[0]))

    # ---- 轨迹统计 ----
    seg = np.hypot(np.diff(traj[:, 0]), np.diff(traj[:, 1]))
    print('  轨迹总长 %.3f m' % seg.sum())
    print('  净位移   %.3f m   （起点→终点直线）'
          % np.hypot(traj[-1, 0] - traj[0, 0], traj[-1, 1] - traj[0, 1]))
    dyaw = np.degrees(np.unwrap(traj[:, 2]))
    print('  净转角   %+.1f °' % (dyaw[-1] - dyaw[0]))
    # ★ 注意：这个比值【只在单任务、且终点≠起点】时才说明"绕路"。
    #   多任务演示里最后一个任务常常就是回到起点 ⇒ 净位移≈0 ⇒ 比值会巨大但正常。
    net = np.hypot(traj[-1, 0] - traj[0, 0], traj[-1, 1] - traj[0, 1])
    if n_res <= 1:
        print('  ★ 轨迹总长 / 净位移 = %.2f   （明显 >1 = 绕路/绕圈）'
              % (seg.sum() / max(net, 1e-6)))
    else:
        print('  （多任务：净位移 %.3f m 不说明问题 —— 最后一个任务可能就是回起点）'
              % net)

    if len(seg) > 3:
        print('  瞬时速度 中位 %.3f m/s  最大 %.3f m/s'
              % (np.median(seg / np.maximum(np.diff(tt), 1e-6)),
                 np.max(seg / np.maximum(np.diff(tt), 1e-6))))

    if 'stage_names' in d.files:
        names = list(d['stage_names'])
        uniq = sorted(set(names))
        print('  出现过的 stage: %s' % uniq)

    # ---- 画图 ----
    fig, ax = plt.subplots(figsize=(10, 8.5))
    if a.grid:
        g = np.load(a.grid)
        occ = g['occ'].astype(bool)
        mo = g['origin']
        res = float(g['res'])
        so = np.array(g['scene_origin']) if 'scene_origin' in g else np.zeros(2)
        ox, oy = float(mo[0] - so[0]), float(mo[1] - so[1])
        H, W = occ.shape
        ax.imshow(occ, origin='lower', cmap='Greys', vmin=0, vmax=1.35,
                  extent=[ox, ox + W * res, oy, oy + H * res], zorder=1)

    for i, p in enumerate(paths):
        if len(p) < 2:
            continue
        ax.plot(p[:, 0], p[:, 1], color='tab:red', lw=1.5, ls='--', alpha=0.75,
                zorder=3, label='planned %d' % i)

    ax.plot(traj[:, 0], traj[:, 1], color='tab:blue', lw=2.2, zorder=4,
            label='travelled')
    # 每 10 个点标一次朝向
    step = max(1, len(traj) // 40)
    ax.quiver(traj[::step, 0], traj[::step, 1],
              np.cos(traj[::step, 2]), np.sin(traj[::step, 2]),
              color='tab:blue', alpha=0.5, zorder=5, width=0.004)
    ax.plot(traj[0, 0], traj[0, 1], 's', color='k', ms=10, zorder=6, label='start')
    ax.plot(traj[-1, 0], traj[-1, 1], 'o', color='lime', ms=12, mec='k',
            zorder=6, label='end')

    # ★ 自动缩放：轨迹常常只占图的一小角（空图是 40m×40m），不缩放根本看不见
    xs = [traj[:, 0].min(), traj[:, 0].max()]
    ys = [traj[:, 1].min(), traj[:, 1].max()]
    for p in paths:
        if len(p):
            xs += [p[:, 0].min(), p[:, 0].max()]
            ys += [p[:, 1].min(), p[:, 1].max()]
    cx, cy = (xs[0] + xs[1]) / 2.0, (ys[0] + ys[1]) / 2.0
    span = max(xs[1] - xs[0], ys[1] - ys[0], 0.5) * 1.35
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)

    ax.set_title('replay: %d pts, %.2f m travelled, %.2f m net   [zoom %.1f m]'
                 % (len(traj), seg.sum(),
                    np.hypot(traj[-1, 0] - traj[0, 0], traj[-1, 1] - traj[0, 1]),
                    span))
    ax.grid(alpha=0.25)
    ax.set_aspect('equal')
    ax.set_xlabel('scene x [m]')
    ax.set_ylabel('scene y [m]')
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(out, dpi=100)
    print('  图已存: %s' % out)

    # ---- 曲线 ----
    if a.curve and len(tt) > 2:
        t = tt - tt[0]
        fig2, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        axs[0].plot(t, s, lw=1.6)
        axs[0].set_ylabel('path length [m]')
        axs[0].grid(alpha=0.25)
        if 'stage_remain' in d.files:
            axs[1].plot(np.array(d['stage_t']) - tt[0], d['stage_remain'], lw=1.6)
            axs[1].set_ylabel('remaining dist [m]')
            axs[1].grid(alpha=0.25)
        axs[2].plot(t, np.degrees(np.unwrap(traj[:, 2])), lw=1.6)
        axs[2].set_ylabel('yaw [deg]')
        axs[2].set_xlabel('t [s]')
        axs[2].grid(alpha=0.25)
        out2 = os.path.splitext(out)[0] + '_curves.png'
        plt.tight_layout()
        plt.savefig(out2, dpi=100)
        print('  曲线已存: %s' % out2)
    print('=' * 60)


if __name__ == '__main__':
    main()
