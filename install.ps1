# LLMZIP Agent Automatic Installer for Windows PowerShell
$ErrorActionPreference = "Stop"

$Repo = "kaohaoyi/LLMZIP"
$InstallDir = if ($env:LLMZIP_INSTALL_DIR) { $env:LLMZIP_INSTALL_DIR } else { "$env:USERPROFILE\.local\bin" }

if (!(Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

$ZipName = "llmzip-v0.1.0-windows-x64.zip"
$Url = "https://github.com/$Repo/releases/download/v0.1.0/$ZipName"
$TempZip = "$env:TEMP\llmzip.zip"

Write-Host "[*] Downloading LLMZIP for Windows x64..." -ForegroundColor Cyan

try {
    Invoke-WebRequest -Uri $Url -OutFile $TempZip -UseBasicParsing
    Expand-Archive -Path $TempZip -DestinationPath $env:TEMP\llmzip_extracted -Force
    Copy-Item -Path "$env:TEMP\llmzip_extracted\llmzip.exe" -Destination "$InstallDir\llmzip.exe" -Force
    Remove-Item $TempZip -Force -ErrorAction SilentlyContinue
    Remove-Item "$env:TEMP\llmzip_extracted" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "[✓] LLMZIP installed successfully to $InstallDir\llmzip.exe" -ForegroundColor Green
} catch {
    Write-Warning "[!] Release binary download failed. Falling back to compiling via cargo git..."
    cargo install --git "https://github.com/$Repo.git" llmzip-cli --locked
    exit $LASTEXITCODE
}

# Verify in PATH
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$InstallDir", "User")
    $env:Path = "$env:Path;$InstallDir"
    Write-Host "[i] Added $InstallDir to User PATH." -ForegroundColor Yellow
}
