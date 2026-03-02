[CmdletBinding()]
param(
    [string]$TemplatePath = "C:\pptx-creator\template\Networld-Basic.pptx"
)

$ErrorActionPreference = "Stop"

if ($PSVersionTable.PSEdition -ne "Core") {
    throw "Run with PowerShell 7+ (pwsh)."
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python was not found in PATH."
}
if (-not (Test-Path -LiteralPath $TemplatePath)) {
    throw "Template not found: $TemplatePath"
}

$root = Split-Path -Parent $PSScriptRoot
$input = Join-Path $root "slide-spec.json"
$output = Join-Path $root "output\AWS_Bedrock_VAST_restructured_20260302.pptx"
$renderer = Join-Path $PSScriptRoot "render_networld_pptx.py"

if (-not (Test-Path -LiteralPath $input)) { throw "Input not found: $input" }
if (-not (Test-Path -LiteralPath $renderer)) { throw "Renderer not found: $renderer" }

python $renderer --template $TemplatePath --input $input --output $output | Out-Null
Write-Output "Created: $output"
