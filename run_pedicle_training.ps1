<#
.SYNOPSIS
    Run LOPO heatmap landmark detector training for AP and LAT pedicle-augmented models.

.DESCRIPTION
    Wraps the Python training script with correct PowerShell error handling.
    The Python/timm backbone initialisation writes a benign warning to stderr
    ("Unexpected keys found while loading pretrained weights") which is caught
    and silenced at the Python level (redirect_stderr in train_landmark_detector.py).
    $ErrorActionPreference = 'Continue' here is a second safety-net so any OTHER
    incidental stderr writes (e.g. CUDA version notices) do not abort the run.

    PREREQUISITE: restore mslevel_reg/data/patients/* and manifests from backup,
    OR regenerate manifests via scripts/generate_synthetic_drrs.py first.
    The script will fail with "manifest not found" if data is missing — that is
    expected and not a bug in the invocation.

.PARAMETER View
    Which view to train: AP, LAT, or Both (default).

.PARAMETER OnlyPatient
    Optional: restrict training to a single LOPO fold (e.g. JYOTHI).

.PARAMETER Epochs
    Number of training epochs per fold (default 200).

.PARAMETER Workers
    DataLoader worker processes (default 8; use 0 for debugging).

.EXAMPLE
    # Train both views, all folds:
    .\run_pedicle_training.ps1

    # Retrain just the JYOTHI LAT fold (most urgently needed):
    .\run_pedicle_training.ps1 -View LAT -OnlyPatient JYOTHI

    # Quick smoke-test (10 epochs, synchronous loading):
    .\run_pedicle_training.ps1 -Epochs 10 -Workers 0
#>

[CmdletBinding()]
param(
    [ValidateSet('AP','LAT','Both')]
    [string]$View = 'Both',

    [string]$OnlyPatient = '',

    [int]$Epochs = 200,

    [int]$Workers = 8
)

Set-StrictMode -Version Latest
# Key fix: do NOT treat native-command stderr as a terminating error.
$ErrorActionPreference = 'Continue'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$BaseCmd = @(
    'python', 'scripts/train_landmark_detector.py',
    '--manifest',  'data/landmark_manifest_real.csv',
    '--epochs',    $Epochs,
    '--batch',     4,
    '--lr',        '3e-4',
    '--device',    'cuda',
    '--workers',   $Workers
)

if ($OnlyPatient -ne '') {
    $BaseCmd += @('--only', $OnlyPatient)
}

function Run-Training {
    param([string]$ViewArg, [string]$SynManifest, [string]$ConvManifest, [string]$CkptDir)

    Write-Host "`n$('='*65)" -ForegroundColor Cyan
    Write-Host "  Training: view=$ViewArg  ckpt=$CkptDir" -ForegroundColor Cyan
    if ($OnlyPatient -ne '') {
        Write-Host "  Fold filter: --only $OnlyPatient" -ForegroundColor Yellow
    }
    Write-Host "$('='*65)" -ForegroundColor Cyan

    $Cmd = $BaseCmd + @(
        '--view',                $ViewArg,
        '--synthetic_manifest',  $SynManifest,
        '--ckpt_dir',            $CkptDir
    )

    # Wire in CycleGAN-converted manifest if it exists (closes domain gap).
    # This was missing from the previous training run (converted=0).
    $ConvPath = Join-Path $ScriptDir $ConvManifest
    if (Test-Path $ConvPath) {
        $Cmd += @('--converted_manifest', $ConvManifest)
        Write-Host "  CycleGAN converted manifest found: $ConvManifest" -ForegroundColor Green
    } else {
        Write-Host "  NOTE: converted manifest not found at $ConvManifest" -ForegroundColor Yellow
        Write-Host "        Run scripts/convert_xrays_to_drr.py first to create it." -ForegroundColor Yellow
    }

    $StartTime = Get-Date
    & $Cmd[0] $Cmd[1..($Cmd.Length-1)]
    $ExitCode = $LASTEXITCODE
    $Elapsed = (Get-Date) - $StartTime

    if ($ExitCode -eq 0) {
        Write-Host "`n  DONE ($ViewArg): elapsed $($Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
    } else {
        Write-Host "`n  ERROR ($ViewArg): exit code $ExitCode after $($Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Red
        Write-Host "  Check that data/landmark_manifest_real.csv and $SynManifest exist." -ForegroundColor Red
    }
    return $ExitCode
}

$ExitCodes = @()

if ($View -eq 'AP' -or $View -eq 'Both') {
    $ExitCodes += Run-Training `
        -ViewArg       'AP' `
        -SynManifest   'data/synthetic_drr_manifest_ap.csv' `
        -ConvManifest  'data/converted_xray_manifest_ap.csv' `
        -CkptDir       'checkpoints/landmark_detector_ap_pedicles'
}

if ($View -eq 'LAT' -or $View -eq 'Both') {
    $ExitCodes += Run-Training `
        -ViewArg       'LAT' `
        -SynManifest   'data/synthetic_drr_manifest_lat.csv' `
        -ConvManifest  'data/converted_xray_manifest_lat.csv' `
        -CkptDir       'checkpoints/landmark_detector_lat_pedicles'
}

Write-Host "`n$('='*65)" -ForegroundColor Cyan
if ($ExitCodes -contains 1) {
    Write-Host "  One or more training runs failed. See output above." -ForegroundColor Red
} else {
    Write-Host "  All training runs completed successfully." -ForegroundColor Green
}
Write-Host "$('='*65)" -ForegroundColor Cyan
