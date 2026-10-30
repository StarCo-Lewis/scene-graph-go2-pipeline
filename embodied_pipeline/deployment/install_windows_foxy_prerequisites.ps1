# Explicit user-run administrator step. No WSL, robot, ROS node or goals.
# Native Foxy uses a separate Python 3.8 ABI, NOT the existing Python 3.12.
[CmdletBinding()]
param(
    [string]$DownloadDirectory = 'C:\Users\1\Documents\Codex\ros2_foxy\downloads',
    [string]$InstallRoot = 'C:\ros2_foxy'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Open PowerShell as Administrator, then run this script. No installation has started.'
}
$resolvedInstall = [IO.Path]::GetFullPath($InstallRoot).TrimEnd('\')
if ($resolvedInstall -ne 'C:\ros2_foxy') {
    throw 'This reviewed prerequisite script uses C:\ros2_foxy; do not redirect it into an existing environment.'
}
$pythonRoot = 'C:\Python38'
$pythonExe = Join-Path $pythonRoot 'python.exe'
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
New-Item -ItemType Directory -Path $DownloadDirectory -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $InstallRoot 'logs') -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$transcript = Join-Path $InstallRoot "logs\prerequisites_$stamp.log"
Start-Transcript -Path $transcript | Out-Null
try {
    Write-Host 'Preparing an isolated ROS2 Foxy client. Existing Python 3.12/WSL stay unchanged.'
    Write-Host 'Python 3.8 and Foxy are legacy/EOL components: use only for this controlled robot integration.'
    Write-Host 'C++ Build Tools may download several GB. No automatic reboot or robot connection.'
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    function Get-VerifiedInstaller([string]$Name, [string]$Url, [string]$ExpectedHash, [string]$Publisher) {
        $path = Join-Path $DownloadDirectory $Name
        if (-not (Test-Path -LiteralPath $path)) {
            Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $path -TimeoutSec 1200
        }
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        if ($hash -ne $ExpectedHash) {
            throw "Installer hash mismatch: $Name. File retained for inspection; no execution."
        }
        $signature = Get-AuthenticodeSignature -LiteralPath $path
        if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notlike "*$Publisher*") {
            throw "Installer signature rejected: $Name"
        }
        Write-Host "Verified $Name ($hash)"
        return $path
    }

    if (Test-Path -LiteralPath $pythonRoot) {
        if (-not (Test-Path -LiteralPath $pythonExe)) {
            throw 'C:\Python38 already exists but is not the expected installation. Refusing to overwrite it.'
        }
        & $pythonExe -c 'import sys, struct; assert sys.version_info[:2] == (3,8) and struct.calcsize(chr(80)) == 8; print(sys.version)'
        if ($LASTEXITCODE -ne 0) { throw 'Existing C:\Python38 is not compatible 64-bit Python 3.8.' }
        Write-Host 'Reusing existing Python 3.8; not reinstalling.'
    } else {
        $installer = Get-VerifiedInstaller 'python-3.8.10-amd64.exe' `
            'https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe' `
            '7628244CB53408B50639D2C1287C659F4E29D3DFDB9084B11AED5870C0C6A48A' 'Python Software Foundation'
        $pythonLog = Join-Path $InstallRoot "logs\python38_$stamp.log"
        $pythonArgs = @('/quiet', 'InstallAllUsers=0', 'TargetDir=C:\Python38',
            'PrependPath=0', 'AssociateFiles=0', 'Shortcuts=0', 'Include_launcher=0',
            'Include_pip=1', 'Include_tcltk=1', 'Include_dev=1', 'Include_test=0',
            'Include_doc=0', '/log', $pythonLog)
        $proc = Start-Process -FilePath $installer -ArgumentList $pythonArgs -PassThru -Wait -WindowStyle Hidden
        if ($proc.ExitCode -notin @(0, 3010)) { throw "Python installation failed: $($proc.ExitCode); see $pythonLog" }
        if ($proc.ExitCode -eq 3010) { Write-Warning 'Python installer requests a reboot; no reboot was performed.' }
    }
    & $pythonExe -c 'import sys, tkinter, venv, ssl; assert sys.version_info[:2] == (3,8); print(sys.executable)'
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.8/Tk import check failed.' }

    $vsPath = $null
    if (Test-Path -LiteralPath $vswhere) {
        $vsPath = & $vswhere -latest -products '*' -version '[16.0,17.0)' `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    }
    if (-not $vsPath) {
        $installer = Get-VerifiedInstaller 'vs_buildtools.exe' `
            'https://aka.ms/vs/16/release/vs_buildtools.exe' `
            '0A641C8F47DF21F3569FBD87F1F2301AE82DB9BC5669BB07B18FB623A207C9BA' 'Microsoft Corporation'
        $buildArgs = @('--passive', '--wait', '--norestart', '--nocache',
            '--installPath', (Join-Path $InstallRoot 'BuildTools'),
            '--add', 'Microsoft.VisualStudio.Workload.VCTools', '--includeRecommended')
        # Visible progress is intentional: this is an interactive, user-run install.
        $proc = Start-Process -FilePath $installer -ArgumentList $buildArgs -PassThru -Wait
        if ($proc.ExitCode -notin @(0, 3010)) { throw "C++ Build Tools installation failed: $($proc.ExitCode)" }
        if ($proc.ExitCode -eq 3010) { Write-Warning 'Build Tools requests a reboot; no reboot was performed.' }
        $vsPath = & $vswhere -latest -products '*' -version '[16.0,17.0)' `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    }
    if (-not $vsPath -or -not (Test-Path -LiteralPath (Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat'))) {
        throw 'Visual Studio 2019 x64 C++ toolchain not found after installation.'
    }
    $record = @{status='prerequisites_ready_ros_not_yet_configured'; python=$pythonExe;
        visual_studio=$vsPath; install_root=$InstallRoot; global_path_modified=$false;
        robot_contacted=$false; ros_interfaces_built=$false; recorded_at=(Get-Date -Format o)}
    $record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $InstallRoot 'prerequisites.json') -Encoding UTF8
    Write-Host ''
    Write-Host 'PREREQUISITES READY. ROS2 and nav_interfaces still need installation/build.'
    Write-Host "Report: $InstallRoot\prerequisites.json"
    Write-Host "Log: $transcript"
} finally {
    Stop-Transcript | Out-Null
}
