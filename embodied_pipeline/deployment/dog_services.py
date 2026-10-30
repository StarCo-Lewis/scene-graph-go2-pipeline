"""Interactive dog-side service launcher, Python 3.8, standard library only.

Uses the supplied 619 SOP. Never sends a navigation goal, deletes pending state,
changes controller settings or patches navigation source automatically.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

WS = Path.home() / 'nav_ws'
BRIDGE = WS / 'nebula_bridge'
FRAME = 'd3_619_slam_world_x_forward_y_left_z_up_m'
PROTOCOL = 'nebula.navigation.tcp.v1'


def process_role(args, cwd):
    """Match known service files, not arbitrary substrings or other robot programs."""
    names = {Path(a).name for a in args}
    if 'tcp_bridge.py' in names and (cwd == str(BRIDGE) or str(BRIDGE/'tcp_bridge.py') in args):
        return 'bridge'
    if ('nav_node_real.py' in names or 'demo.nav_node_real' in args) and (cwd.startswith(str(WS)+'/') or any(str(WS) in a for a in args)):
        return 'navigation'
    if 'live_bridge.py' in names and (cwd == str(WS) or str(WS/'live_bridge.py') in args):
        return 'capo_bridge'
    if 'fusion_estimator_node' in names and any(str(WS/'install') in a for a in args):
        return 'capo'
    return None


def processes():
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid != os.getuid():continue
            args = (p/'cmdline').read_bytes().decode().split('\0')
            role = process_role(args, str((p/'cwd').resolve()))
            if role:found.append((int(p.name), role))
        except (OSError, UnicodeError):pass
    return found


def greeting():
    token = (BRIDGE/'.token').read_text().strip()
    if not token:raise RuntimeError('Empty bridge token')
    with socket.create_connection(('192.168.123.18',8888), timeout=3) as s:
        s.settimeout(3)
        s.sendall((json.dumps({'protocol':PROTOCOL,'type':'hello','token':token})+'\n').encode())
        data=b''
        while not data.endswith(b'\n') and len(data)<65536:
            chunk=s.recv(1)
            if not chunk:raise RuntimeError('Bridge closed before greeting')
            data+=chunk
        row=json.loads(data)
    if row.get('protocol')!=PROTOCOL or row.get('type')!='hello' or row.get('coordinate_frame')!=FRAME:
        raise RuntimeError('Unexpected bridge/frame; refusing automatic management')
    return row


def report():
    print('Service processes:',processes(),flush=True)
    try:
        row=greeting()
        print('Bridge: action_available=%s motion_enabled=%s busy=%s' %
              tuple(row.get(k) for k in ('action_available','motion_enabled','busy')),flush=True)
    except Exception as exc:print('Bridge check:',type(exc).__name__,str(exc),flush=True)


def stop_roles(roles):
    targets=[(pid,role) for pid,role in processes() if role in roles]
    for pid,role in targets:
        # Recheck PID identity before signalling; no broad pkill or forced SIGKILL.
        if (pid,role) in processes():os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+15
    while any(pair in processes() for pair in targets):
        if time.monotonic()>deadline:raise RuntimeError('Service did not exit; inspect manually. No forced kill.')
        time.sleep(.5)


def launch(command, log, cwd=WS, extra_env=None):
    env=os.environ.copy();env['ROS_DOMAIN_ID']='0';env['PYTHONUNBUFFERED']='1'
    if extra_env:env.update(extra_env)
    with log.open('ab') as f:
        return subprocess.Popen(command,cwd=str(cwd),env=env,stdin=subprocess.DEVNULL,
                                stdout=f,stderr=subprocess.STDOUT,start_new_session=True)


def wait_log(log, marker, seconds):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        text=log.read_text(errors='replace') if log.exists() else ''
        if marker in text:return
        time.sleep(1)
    raise RuntimeError('Readiness not confirmed; inspect '+str(log))


def wait_bridge(motion, seconds=25):
    deadline=time.monotonic()+seconds;error=''
    while time.monotonic()<deadline:
        try:
            row=greeting()
            if row.get('busy'):raise RuntimeError('Bridge unexpectedly busy')
            if row.get('action_available') is True and row.get('motion_enabled') is motion:return
            error='Action not ready / wrong bridge mode'
        except Exception as exc:error=str(exc)
        time.sleep(1)
    raise RuntimeError('Bridge readiness failed: '+error)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['start','restart-nav','restart-all','status','patch-4cm'])
    ap.add_argument('--utc',help='Laptop UTC time, only applied before a cold/full start')
    args=ap.parse_args(argv)
    if args.mode=='status':report();return
    import fcntl
    lock=(WS/'.pipeline_services.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    needed=[WS/'start_capo.sh',WS/'start_nav_real.sh',WS/'dogenv_run.sh',
            WS/'nav/tools/619_map_v1_clean_5cm.npz',BRIDGE/'tcp_bridge.py',BRIDGE/'.token']
    for p in needed:
        if not p.is_file():raise RuntimeError('Missing required file: '+str(p))
    if (BRIDGE/'runtime/pending_goal.json').exists():
        raise RuntimeError('Unresolved runtime/pending_goal.json: confirm previous task manually. File preserved.')
    current=processes()
    if any(role=='bridge' for _,role in current):
        if greeting().get('busy'):raise RuntimeError('Navigation active: finish/cancel it before restarting.')
    if args.mode=='start' and current:
        report();print('Existing services preserved. Choose restart only when idle.');return
    print('This will restart services / invalidate navigation alignment. No goal is sent.')
    if input('Confirm robot stopped, client queue stopped, remote controller in hand. Type STOPPED: ').strip()!='STOPPED':
        print('Cancelled; no services changed.');return
    if (BRIDGE/'runtime/pending_goal.json').exists():
        raise RuntimeError('Task state changed while waiting; no restart performed.')
    if any(role=='bridge' for _,role in processes()) and greeting().get('busy'):
        raise RuntimeError('Navigation became active; no restart performed.')
    if args.mode=='patch-4cm':
        if any(role=='navigation' for _,role in current):
            raise RuntimeError('Stop the navigation node before patching; source left unchanged.')
        subprocess.run(['python3',str(Path(__file__).with_name('configure_fine_stop_4cm.py')),
                        str(WS/'nav/control/fine_approach.py'),'--apply'],check=True)
        return
    if args.mode=='restart-nav' and not {'capo','capo_bridge'}.issubset({r for _,r in current}):
        raise RuntimeError('CAPO is not fully running; use full restart after inspection.')
    stop_roles({'bridge','navigation'} if args.mode=='restart-nav' else {'bridge','navigation','capo','capo_bridge'})
    run=WS/'pipeline_service_logs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');run.mkdir(parents=True)
    print('Logs:',run,flush=True)
    (run/'operation.json').write_text(json.dumps({'mode':args.mode,'no_goal_sent':True,'alignment_invalidated':True},indent=2))
    if args.mode!='restart-nav':
        if args.utc:
            datetime.strptime(args.utc,'%Y-%m-%d %H:%M:%S')
            subprocess.run(['sudo','date','-u','-s',args.utc],check=True)  # Password remains in terminal.
        launch(['bash',str(WS/'start_capo.sh')],run/'capo.log')
        wait_log(run/'capo.log','RESULT',90)
        if not {'capo','capo_bridge'}.issubset({r for _,r in processes()}):raise RuntimeError('CAPO processes missing')
    launch(['bash',str(WS/'start_nav_real.sh'),'--action','--capo-scale','1.048',
            '--grid',str(WS/'nav/tools/619_map_v1_clean_5cm.npz'),'--aim-mode','waypoint'],run/'navigation.log')
    wait_log(run/'navigation.log','ActionServer',60)
    # Wait for an old listener/TIME_WAIT to release before starting a new instance.
    deadline=time.monotonic()+70
    while True:
        try:
            with socket.socket() as s:s.bind(('192.168.123.18',8888))
            break
        except OSError:
            if time.monotonic()>deadline:raise RuntimeError('Port 8888 still occupied; inspect manually')
            time.sleep(2)
    token=(BRIDGE/'.token').read_text().strip()
    cmd=['bash',str(WS/'dogenv_run.sh'),'./tcp_bridge.py','--bind','192.168.123.18','--port','8888',
         '--scene-frame',FRAME,'--runtime-dir','./runtime']
    launch(cmd,run/'bridge_readonly.log',BRIDGE,{'FIELD_BRIDGE_TOKEN':token})
    wait_bridge(False)
    print('Read-only handshake passed. Switching bridge to motion-enabled; no goal is sent.',flush=True)
    stop_roles({'bridge'})
    # The old bridge may not enable SO_REUSEADDR; allow its sockets to expire.
    deadline=time.monotonic()+70
    while True:
        try:
            with socket.socket() as s:s.bind(('192.168.123.18',8888))
            break
        except OSError:
            if time.monotonic()>deadline:raise RuntimeError('8888 not released after read-only check')
            time.sleep(2)
    launch(cmd+['--allow-motion'],run/'bridge_motion.log',BRIDGE,{'FIELD_BRIDGE_TOKEN':token})
    wait_bridge(True);report()
    print('READY. Reconnect client and remeasure initial scene pose. Never reuse a stale origin after restart.')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('STOPPED:',str(exc),flush=True)
        print('No automatic retry/goal. Services already started may remain; inspect logs.')
        raise SystemExit(1)
