param(
  [Parameter(Mandatory = $true)][string]$Path
)

# Read all stdin
$raw = [Console]::In.ReadToEnd()
if ([string]::IsNullOrWhiteSpace($raw)) {
  Write-Error "No input. Usage: Get-Clipboard -Raw | powershell -File tools\writefile.ps1 <path>"
  exit 1
}

# Normalize line endings to LF
$content = $raw -replace "`r`n", "`n"
$content = $content -replace "`r", "`n"

# Ensure final newline
if (-not $content.EndsWith("`n")) { $content += "`n" }

# Ensure parent dir
$parent = Split-Path -Parent $Path
if ($parent -and -not (Test-Path $parent)) {
  New-Item -ItemType Directory -Force -Path $parent | Out-Null
}

# Write as UTF-8 without BOM
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($Path, $content, $utf8NoBom)

Write-Host "Wrote: $Path"
(Get-FileHash -Algorithm SHA256 $Path).Hash
