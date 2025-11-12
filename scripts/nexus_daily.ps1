param(
    [string]$VenvActivate = ".\.venv\Scripts\Activate.ps1",
    [string]$LogFile = "logs\\nexus_daily.log"
)

$ErrorActionPreference = "Stop"

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path "$scriptPath\.."
Set-Location $projectRoot

if (Test-Path $VenvActivate) {
    . $VenvActivate
}

if (-not (Test-Path (Split-Path $LogFile))) {
    New-Item -ItemType Directory -Path (Split-Path $LogFile) -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$timestamp] Starting Nexus daily run" | Out-File -FilePath $LogFile -Encoding UTF8 -Append

python -m scripts.nexus_daily 2>&1 | Tee-Object -FilePath $LogFile -Encoding UTF8 -Append

$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    "[$(Get-Date -Format "yyyy-MM-dd HH:mm:ss")] Nexus daily run failed with code $exitCode" | Out-File -FilePath $LogFile -Encoding UTF8 -Append
}

exit $exitCode
