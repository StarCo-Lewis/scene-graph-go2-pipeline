# Explicit user-run step 2. Does not contact a robot or change global PATH.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$root = 'C:\ros2_foxy'
$basePython = 'C:\Python38\python.exe'
$helper = Join-Path $PSScriptRoot 'windows_foxy_runtime.py'
if (-not (Test-Path -LiteralPath "$root\prerequisites.json")) {
    throw 'Run install_windows_foxy_prerequisites.ps1 successfully first.'
}
if (-not (Test-Path -LiteralPath $basePython)) { throw 'Python 3.8 not found.' }
$log = Join-Path $root ('logs\runtime_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
Start-Transcript -Path $log | Out-Null
try {
    Write-Host 'Installing ROS2 runtime in C:\ros2_foxy. No WSL or robot connection.'
    Write-Host 'Existing Python 3.12 and machine/user PATH remain unchanged.'
    & $basePython -u -B $helper install
    if ($LASTEXITCODE -ne 0) {
        throw "Runtime setup is incomplete. Keep the output above for diagnosis. Log: $log"
    }
    Write-Host 'ROS2 RUNTIME IMPORTS READY. nav_interfaces still requires compilation.'
    Write-Host "Log: $log"
} finally {
    Stop-Transcript | Out-Null
}
