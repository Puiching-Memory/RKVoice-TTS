param(
    [string]$OutputRoot = "",
    [string]$PackageName = "rk3588-tts-delivery",
    [string]$Version = "",
    [string]$ReleaseNotesPath = "",
    [switch]$IncludeRuntimeBundle,
    [switch]$IncludeEvidence
)

$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $scriptRoot)

if (-not $OutputRoot) {
    $OutputRoot = Join-Path $workspaceRoot 'artifacts\releases'
}

function Get-SafePathSegment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $sanitized = $Value.Trim()
    if (-not $sanitized) {
        return ''
    }

    foreach ($invalidChar in [System.IO.Path]::GetInvalidFileNameChars()) {
        $sanitized = $sanitized.Replace([string]$invalidChar, '-')
    }

    $sanitized = [System.Text.RegularExpressions.Regex]::Replace($sanitized, '\s+', '-')
    return $sanitized.Trim('-')
}

function Copy-WorkspaceItem {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $source = Join-Path $workspaceRoot $RelativePath
    if (-not (Test-Path $source)) {
        return $false
    }

    $destination = Join-Path $releaseDir $RelativePath
    $destinationParent = Split-Path -Parent $destination
    if ($destinationParent) {
        New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null
    }

    Copy-Item -LiteralPath $source -Destination $destination -Recurse -Force
    return $true
}

function Copy-RequiredWorkspaceItem {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    if (-not (Copy-WorkspaceItem -RelativePath $RelativePath)) {
        throw "发布内容缺失：$RelativePath"
    }
}

function Resolve-ReleaseNotesSource {
    if ($ReleaseNotesPath) {
        return (Resolve-Path -LiteralPath $ReleaseNotesPath -ErrorAction Stop).Path
    }

    $defaultTemplate = Join-Path $workspaceRoot 'docs\delivery\发布说明模板.md'
    if (-not (Test-Path $defaultTemplate)) {
        throw '缺少默认发布说明模板：docs\\delivery\\发布说明模板.md'
    }

    return $defaultTemplate
}

function New-ReleaseNotes {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourcePath,
        [Parameter(Mandatory = $true)]
        [string]$DestinationPath,
        [Parameter(Mandatory = $true)]
        [string]$ResolvedVersion,
        [Parameter(Mandatory = $true)]
        [string]$BuildTimestamp,
        [Parameter(Mandatory = $true)]
        [string]$GeneratedAt,
        [Parameter(Mandatory = $true)]
        [string]$ResolvedReleaseDir,
        [Parameter(Mandatory = $true)]
        [string]$ResolvedZipPath
    )

    $content = Get-Content -LiteralPath $SourcePath -Raw -Encoding UTF8
    $rendered = $content.
        Replace('{{PACKAGE_NAME}}', $PackageName).
        Replace('{{VERSION}}', $ResolvedVersion).
        Replace('{{BUILD_TIMESTAMP}}', $BuildTimestamp).
        Replace('{{GENERATED_AT}}', $GeneratedAt).
        Replace('{{RELEASE_DIRECTORY}}', $ResolvedReleaseDir).
        Replace('{{ZIP_PATH}}', $ResolvedZipPath)

    $rendered | Set-Content -LiteralPath $DestinationPath -Encoding UTF8
}

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$generatedAt = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
$resolvedVersion = if ($Version) { $Version.Trim() } else { 'snapshot' }
$packageLabel = Get-SafePathSegment -Value $PackageName
$versionLabel = Get-SafePathSegment -Value $resolvedVersion

if (-not $packageLabel) {
    throw 'PackageName 不能为空，也不能只包含非法文件名字符。'
}

if (-not $versionLabel) {
    throw 'Version 不能为空，也不能只包含非法文件名字符。'
}

$releaseDir = Join-Path $OutputRoot "$packageLabel-$versionLabel-$timestamp"
$zipPath = "$releaseDir.zip"

New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null

$baseItems = @(
    '.gitignore',
    'README.md',
    'pyproject.toml',
    'uv.lock',
    'config\examples',
    'docs',
    'scripts'
)

$includedItems = [System.Collections.Generic.List[string]]::new()

foreach ($item in $baseItems) {
    Copy-RequiredWorkspaceItem -RelativePath $item
    [void]$includedItems.Add($item)
}

$releaseNotesSource = Resolve-ReleaseNotesSource
$releaseNotesTarget = Join-Path $releaseDir 'RELEASE_NOTES.md'
New-ReleaseNotes -SourcePath $releaseNotesSource -DestinationPath $releaseNotesTarget -ResolvedVersion $resolvedVersion -BuildTimestamp $timestamp -GeneratedAt $generatedAt -ResolvedReleaseDir $releaseDir -ResolvedZipPath $zipPath
[void]$includedItems.Add('RELEASE_NOTES.md')

if ($IncludeRuntimeBundle) {
    $runtimeBundle = 'artifacts\runtime\paddlespeech_tts_armlinux_runtime.tar.gz'
    Copy-RequiredWorkspaceItem -RelativePath $runtimeBundle
    [void]$includedItems.Add($runtimeBundle)
}

if ($IncludeEvidence) {
    $evidenceDir = 'artifacts\runtime\paddlespeech_tts_armlinux_runtime\output'
    Copy-RequiredWorkspaceItem -RelativePath $evidenceDir
    [void]$includedItems.Add($evidenceDir)
}

$manifestLines = @(
    "# Release Manifest",
    "",
    "GeneratedAt=$generatedAt",
    "PackageName=$PackageName",
    "PackageLabel=$packageLabel",
    "Version=$resolvedVersion",
    "BuildTimestamp=$timestamp",
    "ReleaseNotesSource=$releaseNotesSource",
    "ReleaseDirectory=$releaseDir",
    "ZipPath=$zipPath",
    "IncludeRuntimeBundle=$($IncludeRuntimeBundle.IsPresent)",
    "IncludeEvidence=$($IncludeEvidence.IsPresent)",
    "",
    "IncludedItems:"
)

foreach ($item in $includedItems) {
    $manifestLines += "- $item"
}

$manifestPath = Join-Path $releaseDir 'RELEASE_MANIFEST.md'
$manifestLines | Set-Content -LiteralPath $manifestPath -Encoding UTF8

if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

Compress-Archive -Path $releaseDir -DestinationPath $zipPath -Force

Write-Host "Release version:   $resolvedVersion"
Write-Host "Release directory: $releaseDir"
Write-Host "Release archive:   $zipPath"
