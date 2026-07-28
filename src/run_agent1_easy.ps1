# run_agent1_easy.ps1
#
# AGENT1_WEIGHT_PATH / AGENT1_LOG_PATH 환경변수를 매번 손으로 안 쳐도 되게,
# 이 스크립트가 자동으로 설정한 뒤 run_agent1_stream.py를 실행해줌.
#
# ----- 사용 전 준비물 (이 .ps1 파일과 "같은 폴더"에 아래 걸 넣어두세요) -----
#   agent1_predictor.py
#   run_agent1_stream.py
#   weights\best_lstm_2d.pth  <- 학습된 가중치 파일 (test_agent1_agent2_connection.py와 경로 통일)
#   preprocessed\             <- X_test.npy, y_test.npy, cutter_test.npy 들어있는 폴더
#
# ----- 실행법 -----
#   .\run_agent1_easy.ps1
#
# 다른 폴더 구조를 쓰고 싶으면 아래처럼 옵션을 직접 지정할 수 있음:
#   .\run_agent1_easy.ps1 -WeightPath "C:\다른경로\best_lstm_2d.pth" -DataDir "C:\다른경로\preprocessed"

param(
    [string]$WeightPath = "$PSScriptRoot\weights\best_lstm_2d.pth",
    [string]$DataDir    = "$PSScriptRoot\preprocessed",
    [string]$LogPath    = "$PSScriptRoot\agent1_stream_log.csv",
    [int]$Limit         = 0,       # 0이면 전체 처리, 숫자 주면 그만큼만
    [double]$Delay      = 0.2
)

$ErrorActionPreference = "Stop"

Write-Host "=== 파일 확인 ===" -ForegroundColor Cyan

if (-not (Test-Path $WeightPath)) {
    Write-Host "가중치 파일을 못 찾았습니다: $WeightPath" -ForegroundColor Red
    Write-Host "best_lstm_2d.pth 파일을 weights\ 폴더 아래에 넣거나, -WeightPath로 경로를 지정하세요." -ForegroundColor Yellow
    exit 1
}
Write-Host "가중치 파일 확인: $WeightPath"

$requiredFiles = @("X_test.npy", "y_test.npy", "cutter_test.npy")
foreach ($f in $requiredFiles) {
    $p = Join-Path $DataDir $f
    if (-not (Test-Path $p)) {
        Write-Host "필요한 데이터 파일이 없습니다: $p" -ForegroundColor Red
        exit 1
    }
}
Write-Host "데이터 폴더 확인: $DataDir"

Write-Host "`n=== 환경변수 자동 설정 ===" -ForegroundColor Cyan
# 이 두 줄이 지난번에 헷갈리셨던 "AGENT1_WEIGHT_PATH"를 손으로 안 쳐도 되게 대신 해주는 부분
$env:AGENT1_WEIGHT_PATH = $WeightPath
$env:AGENT1_LOG_PATH = $LogPath
Write-Host "AGENT1_WEIGHT_PATH = $env:AGENT1_WEIGHT_PATH"
Write-Host "AGENT1_LOG_PATH    = $env:AGENT1_LOG_PATH"

Write-Host "`n=== Agent 1 스트리밍 실행 ===" -ForegroundColor Cyan
$argsList = @("--data-dir", $DataDir, "--delay", $Delay)
if ($Limit -gt 0) {
    $argsList += @("--limit", $Limit)
}

python "$PSScriptRoot\run_agent1_stream.py" @argsList

Write-Host "`n=== 완료 ===" -ForegroundColor Green
Write-Host "다른 터미널 창에서 아래 명령어로 대시보드를 띄우세요:"
Write-Host "  cd `"$PSScriptRoot`"" -ForegroundColor Yellow
Write-Host "  `$env:AGENT1_LOG_PATH = `"$LogPath`"" -ForegroundColor Yellow
Write-Host "  streamlit run streamlit_dashboard.py" -ForegroundColor Yellow