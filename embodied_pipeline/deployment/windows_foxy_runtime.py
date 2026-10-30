"""User-run isolated Windows Foxy setup and import-only diagnostics.

No rclpy.init, DDS discovery, navigation goals, registry edits or global PATH.
Hashes pin the downloaded artifacts; they are not publisher-provided signatures.
"""
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path('C:/ros2_foxy')
DOWNLOADS = Path('C:/Users/1/Documents/Codex/ros2_foxy/downloads')
ROS_ARCHIVE = 'ros2-foxy-20230620-windows-release-amd64.zip'
ROS_HASH = '06cff31303f399f6934c1f9eb2ac7c3d6a31b6bf6c1e0e18d4dde11d96056345'
XML_HASH = 'e3a06a56221f1d2c6ad92b82b416936073468038d13479a65fd39b042f2f4f49'


def extract_checked(archive, destination, expected_hash):
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected_hash:
        raise ValueError('Archive hash mismatch: ' + str(archive))
    marker = destination / '.nebula_archive_sha256'
    if destination.exists():
        if marker.is_file() and marker.read_text().strip() == expected_hash:
            print('Reusing verified extraction:', destination, flush=True)
            return
        raise ValueError('Existing/uncompleted directory retained, refusing overwrite: ' + str(destination))
    with zipfile.ZipFile(archive) as bundle:
        base = destination.resolve()
        for item in bundle.infolist():
            target = (base / item.filename).resolve()
            try:
                target.relative_to(base)
            except ValueError:
                raise ValueError('Unsafe archive member: ' + item.filename)
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive symlinks are not accepted')
        if bundle.testzip() is not None:
            raise ValueError('ZIP CRC validation failed')
        destination.mkdir(parents=True)
        items = bundle.infolist()
        for index, item in enumerate(items, 1):
            bundle.extract(item, destination)
            if index % 2000 == 0 or index == len(items):
                print('Extracted {}/{}: {}'.format(index, len(items), archive.name), flush=True)
    marker.write_text(expected_hash + '\n', encoding='ascii')


def check():
    # Imports/type-support loading only: this does NOT initialize a ROS node.
    errors = []
    dll_search_handles = []
    for folder in (ROOT / 'runtime/ros2-windows/bin', ROOT / 'deps/tinyxml2/lib',
                   Path('C:/Program Files/OpenSSL-Win64/bin')):
        if folder.is_dir():
            dll_search_handles.append(os.add_dll_directory(str(folder)))
    for name in ('tkinter', 'rclpy', 'rosidl_runtime_py.convert', 'ros2cli.cli'):
        try:
            mod = importlib.import_module(name)
            print('IMPORT OK:', name, getattr(mod, '__file__', ''), flush=True)
        except Exception as exc:
            errors.append('{}: {!r}'.format(name, exc))
    ros = ROOT / 'runtime/ros2-windows'
    handles = []
    for name in ('rcl.dll', 'rmw_fastrtps_cpp.dll'):
        try:
            handles.append(ctypes.WinDLL(str(ros / 'bin' / name)))
            print('NATIVE LOAD OK:', name, flush=True)
        except OSError as exc:
            errors.append('{}: {!r}'.format(name, exc))
    try:
        from rclpy.type_support import check_for_type_support
        from example_interfaces.action import Fibonacci
        check_for_type_support(Fibonacci)
        print('BUILT-IN ACTION TYPE SUPPORT OK', flush=True)
    except Exception as exc:
        errors.append('Built-in action native support: {!r}'.format(exc))
    report = {'status': 'runtime_imports_ready_interfaces_not_built' if not errors else 'runtime_dependencies_incomplete',
              'python': sys.executable, 'errors': errors,
              'robot_contacted': False, 'nav_interfaces_built': False}
    print(json.dumps(report, indent=2), flush=True)
    (ROOT / 'runtime_status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 2 if errors else 0


def install():
    if sys.platform != 'win32' or sys.version_info[:2] != (3, 8):
        raise ValueError('Use C:\\Python38\\python.exe on Windows')
    prereq = json.loads((ROOT / 'prerequisites.json').read_text(encoding='utf-8-sig'))
    if prereq.get('status') != 'prerequisites_ready_ros_not_yet_configured':
        raise ValueError('Prerequisite installation has not passed')
    extract_checked(DOWNLOADS / ROS_ARCHIVE, ROOT / 'runtime', ROS_HASH)
    extract_checked(DOWNLOADS / 'tinyxml2.6.0.0.nupkg', ROOT / 'deps/tinyxml2', XML_HASH)
    venv = ROOT / 'client_env'
    python = venv / 'Scripts/python.exe'
    if not venv.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
    if not python.is_file():
        raise ValueError('Incomplete virtual environment retained: ' + str(venv))
    env = os.environ.copy()
    env['PYTHONNOUSERSITE'] = '1'
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    pip = [str(python), '-m', 'pip', '--disable-pip-version-check']
    print('Installing Python dependencies into the isolated client_env...', flush=True)
    subprocess.run(pip + ['install', 'pip==24.3.1', 'setuptools==59.6.0', 'wheel==0.45.1'], check=True, env=env)
    subprocess.run(pip + ['install', 'setuptools==59.6.0', 'empy==3.3.4', 'numpy==1.24.4', 'lark-parser==0.12.0',
                         'PyYAML==6.0.2', 'catkin-pkg==1.0.0', 'packaging==24.2',
                         'cmake==3.27.9', 'colcon-common-extensions', 'netifaces==0.11.0',
                         'importlib-metadata==8.5.0', 'cryptography==44.0.3'], check=True, env=env)
    # Process-local settings only; no machine/user environment changes.
    env['PATH'] = ';'.join([str(venv / 'Scripts'), str(ROOT / 'deps/tinyxml2/lib'),
                          str(ROOT / 'runtime/ros2-windows/bin'), env.get('PATH', '')])
    env['COLCON_PYTHON_EXECUTABLE'] = str(python)
    env['ROS_DOMAIN_ID'] = '0'
    helper = str(Path(__file__).resolve())
    # Do not start a shell with robot commands; setup only establishes search paths.
    command = 'call "C:\\ros2_foxy\\runtime\\ros2-windows\\local_setup.bat" && "{}" -u -B "{}" check'.format(python, helper)
    return subprocess.run(['cmd.exe', '/d', '/s', '/c', command], env=env).returncode


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('install', 'check'):
        raise SystemExit('Usage: windows_foxy_runtime.py install|check')
    raise SystemExit(install() if sys.argv[1] == 'install' else check())
