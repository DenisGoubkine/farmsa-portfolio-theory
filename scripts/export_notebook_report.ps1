param(
    [string]$Notebook = "M4_fama_french.ipynb",
    [string]$OutputBase = "M4_fama_french_report"
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ([System.IO.Path]::IsPathRooted($Notebook)) {
    $notebookPath = (Resolve-Path $Notebook).Path
} else {
    $notebookPath = (Resolve-Path (Join-Path $projectRoot $Notebook)).Path
}

$notebookDir = Split-Path -Parent $notebookPath
$htmlPath = Join-Path $notebookDir ($OutputBase + ".html")
$pdfPath = Join-Path $notebookDir ($OutputBase + ".pdf")

Write-Host "Executing notebook and exporting HTML..."
jupyter nbconvert --to html --execute $notebookPath --output $OutputBase --output-dir $notebookDir
if ($LASTEXITCODE -ne 0) {
    throw "nbconvert failed with exit code $LASTEXITCODE"
}

$browserCandidates = @(
    "C:\Program Files\Google\Chrome\Application\chrome.exe",
    "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
)

$browser = $browserCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $browser) {
    throw "Could not find Chrome or Edge executable for PDF export."
}

$htmlUri = "file:///" + ($htmlPath -replace "\\", "/")

Write-Host "Rendering PDF with headless browser..."
& $browser --headless=new --disable-gpu --allow-file-access-from-files --no-pdf-header-footer "--print-to-pdf=$pdfPath" $htmlUri
if ($LASTEXITCODE -ne 0) {
    throw "Browser PDF rendering failed with exit code $LASTEXITCODE"
}

$maxWaitSeconds = 15
$elapsed = 0
while (-not (Test-Path $pdfPath) -and $elapsed -lt $maxWaitSeconds) {
    Start-Sleep -Seconds 1
    $elapsed += 1
}

if (-not (Test-Path $pdfPath)) {
    throw "PDF file was not created at $pdfPath"
}

$pdf = Get-Item $pdfPath
Write-Host ("Done: " + $pdf.FullName + " (" + $pdf.Length + " bytes)")
