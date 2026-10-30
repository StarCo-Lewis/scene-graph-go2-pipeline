"""Read-only navigation evidence and 2-D occupancy rendering, standard library only.

NPY subset deliberately excludes pickle/object arrays. Outputs interoperate with
the supplied replay_view.py without requiring NumPy/ROS/Pillow in the Tk client.
"""
import ast
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import struct
import time
import zipfile
import zlib

from .field_backend import feedback_pose
from ..execution.navigation_session import terminal_pose
from ..execution.tcp_protocol import clean
from ..execution.navigation_outcome import outcome

DEFAULT_GRID = Path(__file__).parent / "assets/navigation_619/619_map_v1_clean_5cm.npz"


def read_npz(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist()) > 100_000_000:
            raise ValueError("NPZ exceeds viewer limit")
        for item in archive.infolist():
            if not item.filename.endswith('.npy'):
                continue
            raw = archive.read(item)
            if raw[:6] != b'\x93NUMPY' or raw[6] not in (1, 2):
                raise ValueError("Unsupported NPY format")
            offset = 10 if raw[6] == 1 else 12
            size = struct.unpack('<H' if offset == 10 else '<I', raw[8:offset])[0]
            header = ast.literal_eval(raw[offset:offset+size].decode('latin1'))
            shape, dtype = header['shape'], header['descr']
            if header['fortran_order'] or not isinstance(shape, tuple):
                raise ValueError("Only C-order arrays are supported")
            count = math.prod(shape)
            body = raw[offset+size:]
            types = {'|u1': 'B', '|b1': '?', '<f8': 'd', '<f4': 'f', '<i8': 'q', '<i4': 'i'}
            if dtype in types:
                fmt = types[dtype]
                if len(body) != count*struct.calcsize(fmt):
                    raise ValueError("Array size mismatch")
                values = list(struct.unpack('<'+str(count)+fmt, body))
            elif dtype.startswith('<U'):
                width = int(dtype[2:])*4
                if len(body) != count*width:
                    raise ValueError("String array size mismatch")
                values = [body[i:i+width].decode('utf-32-le').rstrip('\0') for i in range(0,len(body),width)]
            else:
                raise ValueError("Unsupported dtype (object/pickle forbidden): " + dtype)
            result[item.filename[:-4]] = (shape, values)
    return result


def npy(values, shape, dtype='<f8'):
    header = repr({'descr': dtype, 'fortran_order': False, 'shape': tuple(shape)}).encode('ascii')
    header += b' ' * ((64-(10+len(header)+1)%64)%64) + b'\n'
    if dtype.startswith('<U'):
        width = int(dtype[2:])*4
        body = b''.join(str(v).encode('utf-32-le').ljust(width,b'\0') for v in values)
    else:
        body = struct.pack('<'+str(len(values))+('q' if dtype == '<i8' else 'd'), *values)
    return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+body


def atomic(path, data):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)


class Grid:
    def __init__(self, path=DEFAULT_GRID):
        self.path = Path(path)
        d = read_npz(path)
        shape, self.occ = d['occ']
        if len(shape) != 2 or not all(isinstance(v,int) and v > 0 for v in shape):
            raise ValueError('Invalid grid shape')
        self.h, self.w = shape
        self.res = d['res'][1][0]
        origin = d['origin'][1]
        so = d.get('scene_origin', ((2,),[0,0]))[1]
        if len(origin) != 2 or len(so) != 2 or not all(math.isfinite(x) for x in origin+so+[self.res]) or self.res <= 0:
            raise ValueError('Invalid grid geometry')
        if any(x not in (0,1) for x in self.occ):
            raise ValueError('Expected binary occupancy')
        self.origin = (origin[0]-so[0],origin[1]-so[1])
        x,y = self.origin
        self.bounds = (x,x+self.w*self.res,y,y+self.h*self.res)
        room = d.get('room_bounds', ((),[]))[1]
        self.view = (room[0]-so[0]-.7,room[1]-so[0]+.7,room[2]-so[1]-.7,room[3]-so[1]+.7) if len(room)==4 else self.bounds
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()


class Evidence:
    def __init__(self, grid, directory=None):
        self.grid, self.directory = grid, Path(directory) if directory else None
        self.traj, self.times, self.segments, self.stages, self.remain = [], [], [], [], []
        self.paths, self.results, self.path_meta = [], [], []
        self.goals, self.task_folders, self.pending_snapshots = [], {}, set()
        self.previews=[]
        self.pose, self.last_pose_time, self.stage = None, None, 'waiting'
        self.expected_frame, self.errors, self.run_dirs = None, [], set()
        self.reported_result_count = None
        self.break_id = 0
        if self.directory:
            self.directory.mkdir(parents=True,exist_ok=False)
            (self.directory/'map.npz').write_bytes(grid.path.read_bytes())

    def consume(self, row, run_dir=None, record=True):
        stamp = row.get('received_utc') or row.get('utc')
        try:
            received = row.get('received_utc')
            t = datetime.fromisoformat(received.replace('Z','+00:00')).timestamp() if received else time.time()
        except (ValueError, AttributeError):
            t = time.time()
        if run_dir:
            self.run_dirs.add(str(run_dir))
        if record and self.directory:
            rid=row.get('request_id')
            if rid:
                if rid not in self.task_folders:
                    folder=self.directory/'tasks'/('task_%03d'%(len(self.task_folders)+1))
                    folder.mkdir(parents=True,exist_ok=True);self.task_folders[rid]=folder
                task_folder=self.task_folders[rid]
                with (task_folder/'events.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(clean({'received_utc':datetime.now(timezone.utc).isoformat(),**row}),ensure_ascii=False,allow_nan=False)+'\n')
                if row.get('type')=='dispatch_reserved':
                    atomic(task_folder/'task.json',json.dumps(clean(row),ensure_ascii=False,indent=2).encode('utf-8'))
                if row.get('type') in ('result','dispatch_rejected','goal_rejected','execution_state_unknown','connection_lost'):
                    atomic(task_folder/'outcome.json',json.dumps(clean({'raw_event':row,'execution_outcome':outcome(row)}),ensure_ascii=False,indent=2).encode('utf-8'))
                    self.pending_snapshots.add(rid)
            with (self.directory/'events.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(clean({'received_utc':datetime.now(timezone.utc).isoformat(),**row}),ensure_ascii=False,allow_nan=False)+'\n')
        kind, d = row.get('type'), row.get('data',{})
        if kind=='candidate_goals':
            self.previews=[]
            for i,item in enumerate(d.get('goals',[])):
                pose=[item.get('goal',{}).get(k) for k in ('x','y','yaw')]
                if not all(not isinstance(x,bool) and isinstance(x,(float,int)) and math.isfinite(x) for x in pose):
                    self.errors.append('Invalid preview goal');continue
                self.previews.append({'label':'P'+str(i+1),'target_element_id':item.get('target_element_id'),
                    'pose':pose,'state':'候选 / 未执行','request_id':None,'audit_context':d.get('audit_context',{}),
                    'bbox_clearance_m':d.get('bbox_clearance_m'),'step_index':item.get('step_index')})
        if kind=='dispatch_reserved':
            g=d.get('goal') or {}
            values=[g.get('g_'+k) for k in ('x','y','yaw')]
            if all(not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) for v in values):
                self.previews=[g for g in self.previews if not (g['target_element_id']==d.get('target_element_id') and g['pose']==values)]
                self.goals.append({'label':'G'+str(len(self.goals)+1),'request_id':row.get('request_id'),
                    'target_element_id':d.get('target_element_id'),'pose':values,
                    'source_action_id':d.get('source_action_id'),'state':'请求已预留 / 尚未确认接收',
                    'audit_context':row.get('audit_context',{})})
        for goal in self.goals:
            if row.get('request_id') and goal['request_id']==row.get('request_id'):
                if kind in ('goal_accepted','result','dispatch_rejected','goal_rejected','execution_state_unknown','connection_lost'):
                    result=outcome(row);goal['outcome']=result
                    goal['state']=result['label_zh']+(' / '+result['reason'] if result['reason'] else '')
        if kind in ('connection_lost','execution_state_unknown'):
            self.break_id += 1
            self.last_pose_time = None
            self.stage = kind
            return
        frame = row.get('coordinate_frame') or d.get('coordinate_frame')
        if frame and self.expected_frame and frame != self.expected_frame:
            self.errors.append('Ignored different coordinate_frame: '+str(frame)); return
        if frame:
            self.expected_frame = frame
        pose = None
        if kind == 'feedback':
            pose = feedback_pose({'event':'feedback','data':d},'scene')
            self.stage = str(d.get('stage','feedback'))
        elif kind == 'result':
            self.results.append(clean(row))
            p = terminal_pose(d)
            if p:
                pose = (p['x'],p['y'],p['yaw'])
            self.stage = str(d.get('result',{}).get('reason','result'))
        elif kind == 'planned_path':
            waypoints = d.get('waypoints')
            if (not isinstance(waypoints,list) or len(waypoints)>10000 or not waypoints or
                any(not isinstance(p,(tuple,list)) or len(p)!=3 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in p) for p in waypoints)):
                self.errors.append('Invalid planned path'); return
            self.paths.append(waypoints)
            self.path_meta.append({'request_id':row.get('request_id'),'utc':stamp,
                                   'association':row.get('association','unverified_topic_association')})
        if pose:
            self.pose, self.last_pose_time = tuple(pose), time.monotonic()
            self.traj.append(self.pose); self.times.append(t)
            self.segments.append(str(row.get('request_id') or 'unknown')+':'+str(self.break_id))
            self.stages.append(self.stage)
            r = d.get('remaining_dist')
            self.remain.append(float(r) if isinstance(r,(int,float)) and math.isfinite(r) else float('nan'))

    def save(self, png_data, terminal=False):
        if not self.directory:
            return
        import io
        buf=io.BytesIO()
        arrays={'traj':npy([v for p in self.traj for v in p],(len(self.traj),3)),
                'traj_t':npy(self.times,(len(self.times),)),
                'n_path':npy([len(self.paths)],(),'<i8'), 'n_result':npy([self.reported_result_count if self.reported_result_count is not None else len(self.results)],(),'<i8'),
                'stage_t':npy(self.times,(len(self.times),)), 'stage_remain':npy(self.remain,(len(self.remain),))}
        for name, values in [('stage_names',self.stages),('segment_ids',self.segments)]:
            arrays[name]=npy(values,(len(values),),'<U'+str(max([len(x) for x in values]+[1])))
        for i,p in enumerate(self.paths):
            arrays['path_'+str(i)]=npy([v for pt in p for v in pt],(len(p),3))
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
            for k,v in arrays.items():z.writestr(k+'.npy',v)
        atomic(self.directory/'trajectory.npz',buf.getvalue())
        atomic(self.directory/'trajectory.png',png_data)
        # SVG adds a readable legend and units without a font/image dependency.
        svg='<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="800"><rect width="100%" height="100%" fill="white"/><text x="20" y="25">619 occupancy (5cm) | blue: measured trajectory | red: observed plan | green: last pose</text><image href="trajectory.png" x="0" y="45" width="1000" height="700"/><text x="20" y="770">X right, Y up, metres. Stage: '+html.escape(self.stage)+'. Semantic-map alignment UNVERIFIED.</text></svg>'
        atomic(self.directory/'trajectory.svg',svg.encode())
        if terminal:
            atomic(self.directory/('terminal_%03d.png'%len(self.results)),png_data)
        manifest={'map_sha256':self.grid.sha256,'map_file':'map.npz','map_type':'2d_binary_occupancy_not_raw_pointcloud',
                  'map_collection_date':'2026-10-12/13 (provided reference image)',
                  'map_origin_scene_xy':self.grid.origin,'resolution_m':self.grid.res,
                  'navigation_frame_declaration':self.expected_frame,'semantic_alignment_verified':False,
                  'timestamps':'client receipt UTC; not synchronized sensor capture times',
                  'trajectory_points':len(self.traj),'planned_paths':len(self.paths),'results':self.results,
                  'navigation_goals':self.goals,'candidate_goals_not_executed':self.previews,'task_logs':{k:str(v.relative_to(self.directory)) for k,v in self.task_folders.items()},
                  'imported_result_count_without_details':self.reported_result_count,
                  'path_associations':self.path_meta,'run_dirs':sorted(self.run_dirs),'validation_errors':self.errors,
                  'updated_utc':datetime.now(timezone.utc).isoformat(),'physical_ground_truth':False}
        atomic(self.directory/'manifest.json',json.dumps(clean(manifest),ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8'))
        for rid in list(self.pending_snapshots):
            import copy
            task=copy.copy(self)
            indexes=[i for i,s in enumerate(self.segments) if s.startswith(rid+':')]
            for key in ('traj','times','segments','stages','remain'):
                setattr(task,key,[getattr(self,key)[i] for i in indexes])
            task.pose=task.traj[-1] if task.traj else None
            task.goals=[g for g in self.goals if g['request_id']==rid]
            task.previews=[]
            task.paths=[p for p,m in zip(self.paths,self.path_meta) if m.get('request_id')==rid]
            folder=self.task_folders[rid]
            atomic(folder/'trajectory.png',Renderer(self.grid).render(task))
            task_data={'request_id':rid,'goals':task.goals,'trajectory':task.traj,'receipt_timestamps':task.times,
                       'segment_ids':task.segments,'planned_paths':task.paths,
                       'map_sha256':self.grid.sha256,'physical_ground_truth':False}
            atomic(folder/'trajectory.json',json.dumps(clean(task_data),indent=2,ensure_ascii=False).encode('utf-8'))
            self.pending_snapshots.remove(rid)


def png(w,h,rgb):
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    raw=b''.join(b'\0'+rgb[y*w*3:(y+1)*w*3] for y in range(h))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')


class Renderer:
    def __init__(self,grid,w=1000,h=700):
        self.grid,self.w,self.h=grid,w,h
        self.cached=None

    def render(self,evidence,full=False):
        w,h,g=self.w,self.h,self.grid
        xmin,xmax,ymin,ymax=g.bounds if full else g.view
        # Include all received observations; never crop an out-of-map trajectory silently.
        pts=evidence.traj+[p for path in evidence.paths for p in path]+[g['pose'] for g in evidence.goals+evidence.previews]
        if pts:
            xmin=min(xmin,min(p[0] for p in pts)-.3);xmax=max(xmax,max(p[0] for p in pts)+.3)
            ymin=min(ymin,min(p[1] for p in pts)-.3);ymax=max(ymax,max(p[1] for p in pts)+.3)
        scale=min((w-60)/(xmax-xmin),(h-60)/(ymax-ymin))
        left=(w-(xmax-xmin)*scale)/2;top=(h-(ymax-ymin)*scale)/2
        def xy(x,y):return (int(left+(x-xmin)*scale),int(top+(ymax-y)*scale))
        key=(xmin,xmax,ymin,ymax,full)
        if self.cached is None or key!=self.cached[0]:
            base=bytearray(b'\xf0\xf2\xf5'*(w*h))
            ox,oy=g.origin
            cols=[math.floor((xmin+(px-left)/scale-ox)/g.res) for px in range(w)]
            for py in range(h):
                row=math.floor((ymax-(py-top)/scale-oy)/g.res)
                if not 0<=row<g.h:continue
                for px,col in enumerate(cols):
                    if 0<=col<g.w:
                        c=b'\x25\x28\x30' if g.occ[row*g.w+col] else b'\xff\xff\xff'
                        idx=(py*w+px)*3;base[idx:idx+3]=c
            self.cached=(key,base)
        rgb=bytearray(self.cached[1])
        def dot(x,y,color,r=1):
            for yy in range(max(0,y-r),min(h,y+r+1)):
                for xx in range(max(0,x-r),min(w,x+r+1)):
                    i=(yy*w+xx)*3;rgb[i:i+3]=bytes(color)
        def line(a,b,color,r=1,dash=False):
            dx,dy=b[0]-a[0],b[1]-a[1];n=max(abs(dx),abs(dy),1)
            for i in range(n+1):
                if not dash or i%12<7:dot(round(a[0]+dx*i/n),round(a[1]+dy*i/n),color,r)
        for path in evidence.paths:
            for a,b in zip(path,path[1:]):line(xy(*a[:2]),xy(*b[:2]),(220,65,65),1,True)
        for i in range(1,len(evidence.traj)):
            if evidence.segments[i]==evidence.segments[i-1]:
                line(xy(*evidence.traj[i-1][:2]),xy(*evidence.traj[i][:2]),(35,100,235),1)
        # Tiny dependency-free raster labels remain visible in exported PNGs.
        glyphs={'G':['01110','10000','10000','10111','10001','10001','01110'],
                '0':['01110','10001','10011','10101','11001','10001','01110'],
                '1':['00100','01100','00100','00100','00100','00100','01110'],
                '2':['01110','10001','00001','00010','00100','01000','11111'],
                '3':['11110','00001','00001','01110','00001','00001','11110'],
                '4':['00010','00110','01010','10010','11111','00010','00010'],
                '5':['11111','10000','10000','11110','00001','00001','11110'],
                '6':['01110','10000','10000','11110','10001','10001','01110'],
                '7':['11111','00001','00010','00100','01000','01000','01000'],
                '8':['01110','10001','10001','01110','10001','10001','01110'],
                '9':['01110','10001','10001','01111','00001','00001','01110']}
        glyphs.update({'P':['11110','10001','10001','11110','10000','10000','10000'],
                       'F':['11111','10000','10000','11110','10000','10000','10000'],
                       'S':['01111','10000','10000','01110','00001','00001','11110'],
                       'C':['01111','10000','10000','10000','10000','10000','01111'],
                       'U':['10001','10001','10001','10001','10001','10001','01110'],
                       '-':['00000','00000','00000','11111','00000','00000','00000']})
        for goal in evidence.previews+evidence.goals:
            x,y=xy(*goal['pose'][:2]);color=(230,125,0) if goal in evidence.goals else (110,110,145)
            line((x-6,y-6),(x+6,y+6),color,1);line((x-6,y+6),(x+6,y-6),color,1)
            result_code=goal.get('outcome',{}).get('code')
            suffix={'succeeded':'-S','failed':'-F','cancelled':'-C','unknown':'-U','rejected':'-F'}.get(result_code,'')
            for c,char in enumerate(goal['label']+suffix):
                for row,bits in enumerate(glyphs.get(char,[])):
                    for col,bit in enumerate(bits):
                        if bit=='1':dot(x+10+c*12+col*2,y-18+row*2,color,1)
        if evidence.traj:dot(*xy(*evidence.traj[0][:2]),(10,10,10),4)
        if evidence.pose:
            x,y,yaw=evidence.pose;a=xy(x,y);b=xy(x+.5*math.cos(yaw),y+.5*math.sin(yaw))
            dot(*a,(0,170,105),5);line(a,b,(0,130,80),2)
            for sign in (-1,1):
                q=xy(x+.5*math.cos(yaw)-.13*math.cos(yaw+sign*.5), y+.5*math.sin(yaw)-.13*math.sin(yaw+sign*.5))
                line(q,b,(0,130,80),1)
        origin=xy(0,0);line((origin[0]-5,origin[1]),(origin[0]+5,origin[1]),(180,40,180),1)
        line((origin[0],origin[1]-5),(origin[0],origin[1]+5),(180,40,180),1)
        line((25,h-18),(25+round(scale),h-18),(0,0,0),2)
        self.transform={'scale_px_per_m':scale,'left':left,'top':top,'xmin':xmin,'ymax':ymax}
        return png(w,h,rgb)
