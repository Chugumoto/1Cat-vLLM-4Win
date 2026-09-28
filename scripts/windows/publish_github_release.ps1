#Requires -Version 5.1
<#
.SYNOPSIS
  Create/update Chugumoto/1Cat-vLLM-4Win (private), push main, upload Windows wheels as a Release.

.NOTES
  Prerequisites:
    - gh auth login (GitHub CLI)
    - Local commit already on main
    - Wheels in dist\*.whl
#>
param(
  [string]$Owner = "Chugumoto",
  [string]$Repo = "1Cat-vLLM-4Win",
  [string]$Tag = "v1.5.1-windows-sm70",
  [string]$Title = "Windows V100/SM70 MVP wheels (cp312, CUDA 12.8)",
  [switch]$Public
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
  $env:Path = "$env:ProgramFiles\GitHub CLI;$env:LOCALAPPDATA\Programs\GitHub CLI;" + $env:Path
}
gh auth status | Out-Null

$visibility = if ($Public) { "--public" } else { "--private" }
$full = "$Owner/$Repo"

$exists = $false
try {
  gh repo view $full --json name -q .name | Out-Null
  $exists = $true
} catch {
  $exists = $false
}

if (-not $exists) {
  Write-Host "Creating $full ($visibility)..."
  gh repo create $full $visibility --source=. --remote=origin --disable-wiki --description "1Cat-vLLM Windows fork for Tesla V100 / SM70 (CUDA 12.8)"
} else {
  Write-Host "Repo exists: $full"
  $remotes = git remote
  if ($remotes -notcontains "origin") {
    git remote add origin "https://github.com/$full.git"
  }
}

Write-Host "Pushing main..."
git push -u origin HEAD:main

$notes = Join-Path $Root "dist\RELEASE_NOTES.md"
if (-not (Test-Path $notes)) {
  @"
Windows V100/SM70 prebuilt wheels (Python 3.12, CUDA 12.8, Torch 2.10).
See README.windows.md for install and build.
"@ | Set-Content -Path $notes -Encoding UTF8
}

$wheels = @(Get-ChildItem (Join-Path $Root "dist\*.whl") -ErrorAction Stop)
if ($wheels.Count -lt 1) { throw "No wheels in dist\" }

$releaseExists = $false
try {
  gh release view $Tag --repo $full | Out-Null
  $releaseExists = $true
} catch {
  $releaseExists = $false
}

if (-not $releaseExists) {
  Write-Host "Creating release $Tag..."
  gh release create $Tag @($wheels.FullName) --repo $full --title $Title --notes-file $notes
} else {
  Write-Host "Uploading assets to existing release $Tag..."
  gh release upload $Tag @($wheels.FullName) --repo $full --clobber
}

Write-Host "Done."
Write-Host "Repo:    https://github.com/$full"
Write-Host "Release: https://github.com/$full/releases/tag/$Tag"
