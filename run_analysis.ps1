[CmdletBinding()]
param(
    [ValidateSet('Full', 'Upstream', 'Core', 'Robustness', 'Smoke')]
    [string]$Mode = 'Full',
    [switch]$SkipUpstream,
    [int]$Seed = 20260622,
    [int]$Bootstrap = 100,
    [int]$ProfileDraws = 500,
    [string]$MnarOR = '0.25,0.5,1,2,4',
    [int]$TraditionalRandomReplicates = 100,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Missing .venv. Create it and install config/requirements-lock.txt before running COMPASS.'
}

$arguments = @(
    (Join-Path $Root 'code\run_analysis.py'),
    '--mode', $Mode.ToLowerInvariant(),
    '--seed', $Seed,
    '--bootstrap', $Bootstrap,
    '--profile-draws', $ProfileDraws,
    '--mnar-or', $MnarOR,
    '--traditional-random-replicates', $TraditionalRandomReplicates
)
if ($SkipUpstream) { $arguments += '--skip-upstream' }
if ($DryRun) { $arguments += '--dry-run' }
& $Python @arguments
exit $LASTEXITCODE
