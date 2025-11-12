Param(
    [string]$Tickers = "ASML.AS,TTE.PA,MC.PA",
    [string]$Period = "2y",
    [string]$Interval = "1d",
    [string]$OutDir = "out",
    [string]$BaseName = "run",
    [switch]$Backtest,
    [switch]$MlEval
)

$ErrorActionPreference = "Stop"

Write-Host "[Nexus] Lancement de l'analyse quotidienne..."

$arguments = @(
    "--tickers", $Tickers,
    "--period", $Period,
    "--interval", $Interval,
    "--score",
    "--report",
    "--save",
    "--out-dir", $OutDir,
    "--base-name", $BaseName,
    "--format", "parquet",
    "--charts-dir", "charts"
)

if ($Backtest) {
    $arguments += @("--bt", "--bt-report")
}

if ($MlEval) {
    $arguments += "--ml-eval"
}

python -m stock_analysis @arguments

Write-Host "[Nexus] Rapport généré. Voir $OutDir pour le rapport Markdown et les graphiques."
Write-Host "[Nexus] Pour générer un résumé personnalisé, utilisez le template templates/report.md.j2."
