$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$dogAddress = 'unitree@192.168.123.18'
try {
    Write-Host 'D3-619 狗端服务管理（不发送导航目标）' -ForegroundColor Cyan
    Write-Host '1. 启动全部服务（已有服务时只显示状态，不重复启动）'
    Write-Host '2. 仅重启导航 + TCP 网桥（保留 CAPO，用于导航代码改动后）'
    Write-Host '3. 全部重启（CAPO + 导航 + TCP 网桥）'
    Write-Host '4. 查看状态（不重启）'
    Write-Host '0. 退出'
    $selection = Read-Host '选择'
    $modes = @{'1'='start'; '2'='restart-nav'; '3'='restart-all'; '4'='status'}
    if ($selection -eq '0') { exit 0 }
    if (-not $modes.ContainsKey($selection)) { throw '无效选项，未连接狗。' }
    $mode = $modes[$selection]
    $sshProgram = (Get-Command ssh.exe -ErrorAction Stop).Source
    $scpProgram = (Get-Command scp.exe -ErrorAction Stop).Source
    Write-Host '上传管理脚本。SSH 密码直接在终端输入，不显示、不保存。'
    & $scpProgram -o ConnectTimeout=10 (Join-Path $PSScriptRoot 'dog_services.py') (Join-Path $PSScriptRoot 'configure_fine_stop_4cm.py') "${dogAddress}:~/"
    if ($LASTEXITCODE -ne 0) { throw '上传失败，未启动或重启服务。' }
    $clockArgument = ''
    if ($mode -in @('start','restart-all')) {
        $utcNow = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm:ss')
        $clockArgument = " --utc '$utcNow'"
    }
    Write-Host '连接狗端。需要 SSH / sudo 密码时请手动输入。'
    Write-Host '重启前会要求输入 STOPPED，表示狗及客户端队列已停止。重启后需重新核对起点。' -ForegroundColor Yellow
    & $sshProgram -tt -o ConnectTimeout=10 $dogAddress "python3 ~/dog_services.py $mode$clockArgument"
    if ($LASTEXITCODE -ne 0) { throw '狗端操作未完成，请查看上方原因。不会自动重试。' }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
} finally {
    [void](Read-Host '按回车关闭此窗口')
}
