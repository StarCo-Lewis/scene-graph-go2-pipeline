"""Interpret GO action feedback; never infer arrival from planning or geometry."""
def outcome(row):
    kind,data=row.get('type'),row.get('data',{})
    result=data.get('result',{})
    status=data.get('ros_goal_status')
    code,label='pending','待执行'
    reason=result.get('reason') or data.get('error') or data.get('reason') or ''
    if kind=='result':
        if status==4 and result.get('success') is True and reason=='arrived':code,label='succeeded','成功'
        elif status==5:code,label='cancelled','已取消'
        elif status in (4,6):code,label='failed','失败'
        else:code,label='unknown','状态未知'
    elif kind in ('goal_rejected','dispatch_rejected'):code,label='rejected','已拒绝 / 未执行'
    elif kind in ('execution_state_unknown','connection_lost'):code,label='unknown','状态未知'
    elif kind in ('goal_accepted','feedback'):code,label='running','执行中'
    return {'code':code,'label_zh':label,'reason':str(reason),'ros_goal_status':status,
            'server_success':result.get('success'),'request_id':row.get('request_id'),
            'basis':'GO ROS terminal/transport feedback; no geometric arrival inference',
            'physical_arrival_independently_verified':False}
