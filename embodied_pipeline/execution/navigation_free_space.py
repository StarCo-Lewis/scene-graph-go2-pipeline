"""Static endpoint screening around a BBOX. No route planning or robot commands."""
import math
from functools import lru_cache

from .navigation_contract import pose
from .navigation_points import bounds, bbox_approach, clearance_value, validate_clearance

# Navigation group's supplied global planner / mission config, not changed here.
FOOTPRINT_RADIUS_M = 0.35
FRAME = 'd3_619_slam_world_x_forward_y_left_z_up_m'
POLICY = 'bbox_offset_ring_144_static_disc_v1'


@lru_cache(maxsize=2)
def _load(path, modified_ns, size):
    from ..ui.navigation_plot import Grid
    return Grid(path)


def scene_grid(scene):
    # The supplied occupancy map is specific to D3-619. Never reuse for another scene.
    meta=scene['metadata']
    if meta.get('scene_id')!='D3-619_v1' or meta.get('coordinate_frame')!=FRAME:
        raise ValueError('当前场景没有配置匹配的导航栅格；禁止仅按 BBOX 生成可发送终点。')
    from ..ui.navigation_plot import DEFAULT_GRID
    stat=DEFAULT_GRID.stat()
    return _load(str(DEFAULT_GRID),stat.st_mtime_ns,stat.st_size)


def point_check(grid, point, radius=FOOTPRINT_RADIUS_M):
    """Check a disc against occupied cell rectangles (not only their centres)."""
    p=pose(point,'navigation_point');x,y=p['x'],p['y']
    if not math.isfinite(radius) or radius<0:raise ValueError('Invalid footprint radius')
    ox,oy=grid.origin;res=grid.res
    if x-radius<ox or y-radius<oy or x+radius>=ox+grid.w*res or y+radius>=oy+grid.h*res:
        return {'free':False,'reason':'outside_map_or_footprint_outside_map'}
    col=math.floor((x-ox)/res);row=math.floor((y-oy)/res)
    if grid.occ[row*grid.w+col]!=0:
        return {'free':False,'reason':'target_cell_occupied','cell':[col,row]}
    for iy in range(max(0,math.floor((y-radius-oy)/res)),min(grid.h,math.floor((y+radius-oy)/res)+1)):
        for ix in range(max(0,math.floor((x-radius-ox)/res)),min(grid.w,math.floor((x+radius-ox)/res)+1)):
            if grid.occ[iy*grid.w+ix]==0:continue
            dx=max(ox+ix*res-x,0,x-(ox+(ix+1)*res))
            dy=max(oy+iy*res-y,0,y-(oy+(iy+1)*res))
            if dx*dx+dy*dy<=radius*radius+1e-12:
                return {'free':False,'reason':'footprint_intersects_obstacle','obstacle_cell':[ix,iy]}
    return {'free':True,'reason':'static_endpoint_disc_clear','cell':[col,row]}


def validate_endpoint(grid, goal):
    check=point_check(grid,goal)
    if not check['free']:
        raise ValueError('导航终点未通过栅格检查：'+check['reason']+
                         '。请重新生成环绕候选点或核对手动终点；不会发送。')
    return {'policy_version':POLICY,'map_sha256':grid.sha256,'map_file':str(grid.path),
            'resolution_m':grid.res,'footprint_radius_m':FOOTPRINT_RADIUS_M,
            'static_endpoint_check':check,'path_reachability_verified':False,
            'map_registration_independently_verified':False}


def candidates(node,start,clearance_m):
    """Sample a rounded rectangle ring: exact distance from BBOX, not its centre.

    Equivalent to outward circular coverage of the rectangle boundary. Keep the
    operator's chosen distance; do not silently move to arbitrarily distant rings.
    """
    d=clearance_value(clearance_m);s=pose(start,'current_start')
    # Preserve the existing ambiguity rejection for a start inside a solid object.
    first=bbox_approach(node,start,d)
    low,high=bounds(node);cx,cy=node['bbox_center'][:2]
    hx,hy=(high[0]-low[0])/2,(high[1]-low[1])/2
    result=[first]
    for i in range(144):
        angle=i*2*math.pi/144;ux,uy=math.cos(angle),math.sin(angle)
        left,right=0.,math.hypot(hx,hy)+d+1e-8
        for _ in range(45):
            r=(left+right)/2
            if math.hypot(max(abs(r*ux)-hx,0),max(abs(r*uy)-hy,0))<d:left=r
            else:right=r
        x,y=cx+right*ux,cy+right*uy
        result.append({'x':x,'y':y,'yaw':math.atan2(cy-y,cx-x)})
    result.sort(key=lambda p: (p['x']-s['x'])**2+(p['y']-s['y'])**2)
    return result


def free_approach(node,start,clearance_m,grid):
    tested=candidates(node,start,clearance_m)
    for candidate in tested:
        if point_check(grid,candidate)['free']:
            validate_clearance(node,candidate,clearance_m)
            return candidate
    raise ValueError(f"目标 ID {node.get('id')}：BBOX 外侧 {clearance_m:g} m 环绕采样的 {len(tested)} 个候选点"
                     f"均未通过导航栅格与 {FOOTPRINT_RADIUS_M:g} m 圆形净空检查。"
                     "请核对地图对齐、调整 BBOX 距离或人工指定合格点；未生成可发送终点。")
