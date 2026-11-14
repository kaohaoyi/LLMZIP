# LLMZIP Agent Automatic Installer for Windows PowerShell
$ErrorActionPreference = "Stop"

$Repo = "kaohaoyi/LLMZIP"
$InstallDir = if ($env:LLMZIP_INSTALL_DIR) { $env:LLMZIP_INSTALL_DIR } else { "$env:USERPROFILE\.local\bin" }

if (!(Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

# Architecture detection
$Tag = if ($env:LLMZIP_VERSION) { $env:LLMZIP_VERSION } else { "v0.2.0" }
$Arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLower()
if ($Arch -eq "arm64" -or $env:PROCESSOR_ARCHITECTURE -eq "ARM64") {
    $TargetArch = "arm64"
    $ZipName = "llmzip-$Tag-windows-arm64.zip"
} else {
    $TargetArch = "x64"
    $ZipName = "llmzip-$Tag-windows-x64.zip"
}

$Url = "https://github.com/$Repo/releases/download/$Tag/$ZipName"
$TempZip = "$env:TEMP\llmzip.zip"

Write-Host "[*] Detected Windows Architecture: $TargetArch" -ForegroundColor Cyan
Write-Host "[*] Downloading LLMZIP ($ZipName)..." -ForegroundColor Cyan

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
