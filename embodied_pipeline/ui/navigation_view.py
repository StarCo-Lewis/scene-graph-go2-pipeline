"""Passive map/trajectory tab. No access to send/cancel or navigation coordinates."""
from datetime import datetime
import json
import math
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import uuid

from .._legacy import ROOT
from .navigation_plot import Grid, Evidence, Renderer, read_npz


class NavigationView(ttk.Frame):
    def __init__(self,parent,console):
        super().__init__(parent,padding=8)
        self.console=console
        self.live=None;self.replay=None;self.dirty=True;self.last_save=0;self.terminal_pending=False;self.live_dirty=False
        self.full=tk.BooleanVar(value=False)
        bar=ttk.Frame(self);bar.pack(fill='x')
        ttk.Button(bar,text='实时观察',command=self.show_live).pack(side='left')
        ttk.Button(bar,text='回放日志 / 轨迹 NPZ',command=self.open_replay).pack(side='left')
        ttk.Button(bar,text='保存轨迹图和数据',command=self.save_now).pack(side='left')
        ttk.Checkbutton(bar,text='完整栅格范围',variable=self.full,command=self.invalidate).pack(side='left')
        ttk.Label(self,text='点云衍生二维栅格｜灰 P：候选终点　橙 G：请求终点（S 成功 / F 失败 / C 取消 / U 未知）　蓝：轨迹　绿：最后位姿').pack(anchor='w')
        ttk.Label(self,text='导航组 619 场景系，X 向右 / Y 向上，单位米；与语义图对齐尚未核实。此页只观察，不修改起点、终点或控制参数。',foreground='#a34e13').pack(anchor='w')
        self.state=tk.StringVar(value='等待导航反馈；没有收到路径时不画模拟路径。')
        ttk.Label(self,textvariable=self.state).pack(anchor='w')
        self.output=tk.StringVar(value='收到首条导航记录后自动创建独立存档目录；每 5 秒及终态保存。')
        ttk.Label(self,textvariable=self.output,wraplength=1100).pack(anchor='w')
        self.goals_table=ttk.Treeview(self,columns=('label','target','x','y','yaw','state'),show='headings',height=3)
        for name,color in [('succeeded','#19753c'),('failed','#b42424'),('cancelled','#985f00'),('unknown','#985f00')]:self.goals_table.tag_configure(name,foreground=color)
        for key,title in [('label','图中标注'),('target','物体 ID'),('x','终点 x / m'),('y','终点 y / m'),('yaw','终点 yaw / rad'),('state','实际执行状态')]:
            self.goals_table.heading(key,text=title);self.goals_table.column(key,width=125 if key!='state' else 280,stretch=True)
        self.goals_table.pack(fill='x')
        self.canvas=tk.Canvas(self,bg='#eef0f4',highlightthickness=0)
        self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<Configure>',lambda e:self.invalidate())
        self.canvas.bind('<Motion>',self.hover)
        self.cursor=tk.StringVar(value='移动鼠标查看导航图坐标；底部黑线 = 1 m。')
        ttk.Label(self,textvariable=self.cursor).pack(anchor='w')
        self.photo=None;self.offset=(0,0);self.error=None
        try:
            self.grid=Grid();self.renderer=Renderer(self.grid)
            self.empty=Evidence(self.grid)
        except Exception as exc:
            self.error=str(exc);self.state.set('地图读取失败：'+str(exc))
        self.after(500,self.tick)

    def new_evidence(self,prefix):
        folder=ROOT/'outputs/real_scene_runs'/(prefix+datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
        return Evidence(self.grid,folder)

    def observe(self,row,run_dir=None):
        if self.error or row.get('type') not in ('candidate_goals','dispatch_reserved','goal_accepted','goal_rejected','dispatch_rejected','planned_path','feedback','result','connection_lost','execution_state_unknown'):
            return
        if self.live is None:
            if row.get('type') in ('connection_lost','execution_state_unknown'):return
            self.live=self.new_evidence('navigation_view_')
        self.live.consume(row,run_dir)
        self.live_dirty=True
        if run_dir:
            link=Path(run_dir,'visualization.json')
            if not link.exists():
                link.write_text(json.dumps({'session_directory':str(self.live.directory),'map_sha256':self.grid.sha256},indent=2),encoding='utf-8')
        self.terminal_pending |= row.get('type')=='result'
        self.dirty=True

    def preview_goals(self,route,clearance_m):
        if self.error:return
        c=self.console
        from ..execution.navigation_free_space import scene_grid, validate_endpoint
        checks=[validate_endpoint(scene_grid(c.scene),item['goal']) for item in route]
        self.observe({'type':'candidate_goals','data':{'goals':route,'bbox_clearance_m':clearance_m,
            'audit_context':{'engineering_plan':str(c.loaded_plan_path),'engineering_plan_sha256':c.loaded_plan_hash,
                             'static_endpoint_checks':checks}},
            'coordinate_frame':c.scene['metadata']['coordinate_frame']})

    def invalidate(self):self.dirty=True

    def show_live(self):
        self.replay=None;self.dirty=True

    def hover(self,event):
        tr=getattr(self.renderer,'transform',None) if not self.error else None
        if tr:
            x=tr['xmin']+(event.x-self.offset[0]-tr['left'])/tr['scale_px_per_m']
            y=tr['ymax']-(event.y-self.offset[1]-tr['top'])/tr['scale_px_per_m']
            self.cursor.set('导航图坐标 x=%.3f m, y=%.3f m；黑色标尺 = 1 m；不回填控制字段。'%(x,y))

    def open_replay(self):
        path=filedialog.askopenfilename(title='选择 tcp_events.jsonl / events.jsonl / live_view 轨迹 NPZ',filetypes=[('导航记录','*.jsonl *.npz')])
        if not path:return
        try:
            self.load_replay(Path(path))
        except Exception as exc:messagebox.showerror('无法回放',str(exc))

    def load_replay(self,path):
        # A saved session carries its exact map, never silently substitute today's map.
        source_grid=Grid(path.parent/'map.npz') if (path.parent/'map.npz').exists() else self.grid
        e=Evidence(source_grid)
        if path.suffix.lower()=='.jsonl':
            rows=[]
            with path.open(encoding='utf-8') as f:
                for n,line in enumerate(f,1):
                    if not line.strip():continue
                    try:
                        row=json.loads(line)
                        if not isinstance(row,dict):raise ValueError('Expected JSON object')
                        rows.append(row)
                    except (ValueError,TypeError,AttributeError) as exc:e.errors.append('line %d: %s'%(n,exc))
            # Old per-run logs omitted the ID on dispatch_reserved. Associate only
            # when the entire source contains exactly one explicit request ID.
            ids={r['request_id'] for r in rows if r.get('request_id')}
            for row in rows:
                if row.get('type')=='dispatch_reserved' and not row.get('request_id') and len(ids)==1:
                    row={**row,'request_id':next(iter(ids)), 'association':'legacy_single_request_log_inferred_not_in_original_row'}
                e.consume(row,record=False)
        else:
            d=read_npz(path);shape,values=d['traj']
            e.reported_result_count=int(d.get('n_result',((),[0]))[1][0])
            if len(shape)!=2 or shape[1]!=3 or any(not math.isfinite(v) for v in values):
                raise ValueError('Expected finite traj (N,3)')
            e.traj=[tuple(values[i:i+3]) for i in range(0,len(values),3)]
            e.times=d.get('traj_t',((0,),[]))[1]
            if len(e.times)!=len(e.traj):raise ValueError('traj_t count mismatch')
            e.segments=d.get('segment_ids',((),['legacy_unspecified']*len(e.traj)))[1]
            e.stages=d.get('stage_names',((),['unknown']*len(e.traj)))[1]
            e.remain=d.get('stage_remain',((),[float('nan')]*len(e.traj)))[1]
            if any(len(v)!=len(e.traj) for v in (e.segments,e.stages,e.remain)):raise ValueError('Trajectory metadata count mismatch')
            for k,(sh,vals) in d.items():
                if k.startswith('path_') and k[5:].isdigit():
                    if len(sh)!=2 or sh[1]!=3 or any(not math.isfinite(v) for v in vals):raise ValueError('Invalid saved path')
                    e.paths.append([vals[i:i+3] for i in range(0,len(vals),3)])
            e.pose=e.traj[-1] if e.traj else None
            e.stage='offline_npz'
        self.replay=e;self.replay_source=str(path);self.dirty=True

    def save_now(self):
        try:
            e=self.replay or self.live
            if not e:return
            if e.directory is None:
                folder=ROOT/'outputs/real_scene_runs'/('navigation_replay_'+datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
                folder.mkdir(parents=True);e.directory=folder
                (folder/'map.npz').write_bytes(e.grid.path.read_bytes())
                source=Path(self.replay_source)
                (folder/('source_record'+source.suffix)).write_bytes(source.read_bytes())
                (folder/'replay_source.json').write_text(json.dumps({'source':self.replay_source,'mode':'offline_replay_not_live'},ensure_ascii=False),encoding='utf-8')
            e.save(Renderer(e.grid).render(e,self.full.get()))
            self.output.set('已保存：'+str(e.directory))
        except Exception as exc:self.output.set('保存失败：'+str(exc))

    def tick(self):
        try:
            if self.error:return
            if self.live and self.live_dirty and (self.terminal_pending or time.monotonic()-self.last_save>=5):
                self.live.save(Renderer(self.grid).render(self.live,self.full.get()),self.terminal_pending)
                self.terminal_pending=False;self.live_dirty=False;self.last_save=time.monotonic()
            e=self.replay or self.live or self.empty
            if self.dirty:
                width=max(300,min(1250,self.canvas.winfo_width()))
                height=max(250,min(800,self.canvas.winfo_height()))
                if self.renderer.grid is not e.grid or (self.renderer.w,self.renderer.h)!=(width,height):self.renderer=Renderer(e.grid,width,height)
                self.photo=tk.PhotoImage(data=self.renderer.render(e,self.full.get()),format='png')
                self.canvas.delete('all');self.offset=(0,0)
                self.canvas.create_image(0,0,image=self.photo,anchor='nw')
                self.canvas.create_text(20,18,anchor='nw',text='X →   Y ↑     m',fill='#995211')
                self.canvas.create_text(25,height-32,text='1 m',anchor='w')
                for item in self.goals_table.get_children():self.goals_table.delete(item)
                for goal in e.goals+e.previews:
                    self.goals_table.insert('','end',values=[goal['label'],goal['target_element_id'],*['%.4f'%v for v in goal['pose']],goal['state']],tags=(goal.get('outcome',{}).get('code','pending'),))
                if self.goals_table.get_children():self.goals_table.see(self.goals_table.get_children()[-1])
                self.dirty=False
            age=time.monotonic()-e.last_pose_time if e.last_pose_time else None
            mode='离线回放（不代表当前位置）' if self.replay else ('等待反馈' if age is None else '反馈 %.1f 秒前%s'%(age,'，已过期；不代表已停止' if age>3 else ''))
            self.state.set('%s｜阶段 %s｜轨迹 %d 点｜规划路径 %s｜数据校验问题 %d'%(mode,e.stage,len(e.traj),str(len(e.paths))+' 条（话题关联未独立核验）' if e.paths else '未收到',len(e.errors)))
            if e.directory:self.output.set('自动存档：'+str(e.directory))
        except Exception as exc:
            self.output.set('可视化错误（不改变控制状态）：'+str(exc))
        finally:self.after(500,self.tick)

    def close(self):
        if self.live:self.live.save(Renderer(self.grid).render(self.live,self.full.get()))
