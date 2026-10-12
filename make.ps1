<#
.SYNOPSIS
    Windows dispatcher for the Makefile targets.

.DESCRIPTION
    GNU make is not present on a stock Windows box, and the Makefile is still
    the right artefact for Docker and CI. This shim runs the same commands
    under the same target names so `.\make.ps1 test` matches `make test`.

    Keep it a thin dispatcher: the moment real logic appears here, the two
    files start to drift.

.EXAMPLE
    .\make.ps1 install
    .\make.ps1 check
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Target = 'help',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

# Deliberately NOT 'Stop'. In PowerShell 5.1 that turns a native command's
# stderr into a terminating error, and uv writes its progress to stderr --
# so a perfectly successful `uv sync` would blow up. Failure is detected
# from $LASTEXITCODE in Invoke-Uv, which is the reliable signal.
$ErrorActionPreference = 'Continue'
Set-Location -Path $PSScriptRoot

function Resolve-Uv {
    # uv may be on PATH, in the per-user scripts dir (where `pip install --user`
    # puts it, which is not on PATH by default), or only importable as a module.
    # Returns an object rather than an array: PowerShell unrolls a single-element
    # array to a scalar string, and indexing that yields its first character.
    $onPath = Get-Command uv -ErrorAction SilentlyContinue
    if ($onPath) {
        return [pscustomobject]@{ Exe = $onPath.Source; PreArgs = @() }
    }

    $userScripts = Join-Path $env:APPDATA 'Python'
    if (Test-Path $userScripts) {
        $candidate = Get-ChildItem -Path $userScripts -Filter 'uv.exe' -Recurse `
            -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($candidate) {
            return [pscustomobject]@{ Exe = $candidate.FullName; PreArgs = @() }
        }
    }

    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($py) {
        return [pscustomobject]@{ Exe = $py.Source; PreArgs = @('-m', 'uv') }
    }

    throw "uv not found. Install it with: pip install uv"
}

$script:Uv = Resolve-Uv

# Which torch build this machine holds. Written by the cpu/gpu targets and
# remembered in a gitignored file, so every command selects the same extra
# and nothing can silently revert the other build.
$script:Torch = if (Test-Path .torch) { (Get-Content .torch -Raw).Trim() } else { 'cpu' }

function Invoke-Uv {
    param([string[]]$Arguments)
    $all = @($script:Uv.PreArgs) + $Arguments
    Write-Host "uv $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $script:Uv.Exe @all
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$targets = [ordered]@{
    'install'      = 'Create the venv and install everything, CPU torch by default'
    'hooks'        = 'Install the pre-commit git hooks'
    'gpu'          = 'Switch to the CUDA torch build, and remember it'
    'cpu'          = 'Switch back to the CPU build (what Docker uses)'
    'test'         = 'Run the full test suite with coverage'
    'test-fast'    = 'Skip anything marked slow, gpu or model'
    'lint'         = 'Lint and check formatting'
    'format'       = 'Apply formatting and safe lint fixes'
    'typecheck'    = 'Static type check'
    'check'        = 'Everything CI runs'
    'prepare-data' = 'Phase 1: convert a source dataset to YOLO format'
    'train'        = 'Phase 2: fine-tune the baseline detector'
    'eval'         = 'Phase 5: violation-level metrics and error breakdown'
    'export'       = 'Phase 6: export to ONNX and verify parity'
    'benchmark'    = 'Phase 7: latency and throughput across backends'
    'serve'        = 'Run the API locally'
    'viewer'       = 'Run the API and open the local debug viewer'
    'clean'        = 'Remove caches and build artefacts'
}

switch ($Target) {
    'help' {
        Write-Host "Usage: .\make.ps1 <target>`n"
        foreach ($name in $targets.Keys) {
            Write-Host ("  {0,-14} {1}" -f $name, $targets[$name])
        }
    }
    'install'      { Invoke-Uv @('sync', '--extra', $script:Torch, '--dev') }
    'hooks'        { Invoke-Uv @('run', '--extra', $script:Torch, 'pre-commit', 'install') }
    'gpu'          {
        Set-Content -Path .torch -Value 'gpu' -NoNewline -Encoding utf8
        $script:Torch = 'gpu'
        Invoke-Uv @('sync', '--extra', 'gpu', '--dev')
        Invoke-Uv @('run', '--extra', 'gpu', 'python', '-c',
            "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())")
    }
    'cpu'          {
        Set-Content -Path .torch -Value 'cpu' -NoNewline -Encoding utf8
        $script:Torch = 'cpu'
        Invoke-Uv @('sync', '--extra', 'cpu', '--dev')
    }
    'test'         { Invoke-Uv (@('run', '--extra', $script:Torch, 'pytest') + $Rest) }
    'test-fast'    { Invoke-Uv @('run', '--extra', $script:Torch, 'pytest', '-m', 'not slow and not gpu and not model') }
    'lint'         {
        Invoke-Uv @('run', '--extra', $script:Torch, 'ruff', 'check', '.')
        Invoke-Uv @('run', '--extra', $script:Torch, 'ruff', 'format', '--check', '.')
    }
    'format'       {
        Invoke-Uv @('run', '--extra', $script:Torch, 'ruff', 'format', '.')
        Invoke-Uv @('run', '--extra', $script:Torch, 'ruff', 'check', '--fix', '.')
    }
    'typecheck'    { Invoke-Uv @('run', '--extra', $script:Torch, 'mypy') }
    'check'        {
        & $PSCommandPath 'lint'
        & $PSCommandPath 'typecheck'
        & $PSCommandPath 'test'
    }
    'prepare-data' { Invoke-Uv (@('run', '--extra', $script:Torch, 'python', 'scripts/prepare_data.py') + $Rest) }
    'train'        { Invoke-Uv (@('run', '--extra', $script:Torch, 'python', 'scripts/train.py') + $Rest) }
    'eval'         { Invoke-Uv (@('run', '--extra', $script:Torch, 'python', 'scripts/evaluate.py') + $Rest) }
    'export'       { Invoke-Uv (@('run', '--extra', $script:Torch, 'python', 'scripts/export_onnx.py') + $Rest) }
    'benchmark'    { Invoke-Uv (@('run', '--extra', $script:Torch, 'python', 'scripts/benchmark.py') + $Rest) }
    'serve'        {
        Invoke-Uv @('run', '--extra', $script:Torch, 'uvicorn', 'ppe.api.app:create_app',
            '--factory', '--reload')
    }
    'viewer'       {
        # Restored afterwards: PowerShell keeps $env: set for the whole session,
        # and leaving PPE_MODEL__WEIGHTS behind made a later `make check` in the
        # same terminal fail two tests that expected a missing model. The suite
        # now isolates itself too, but a task runner should not quietly rewrite
        # the shell it was called from.
        $previous = $env:PPE_MODEL__WEIGHTS
        $env:PPE_MODEL__WEIGHTS = 'runs/train/baseline/weights/best.pt'
        Write-Host 'debug viewer: http://127.0.0.1:8000' -ForegroundColor Cyan
        try {
            Invoke-Uv @('run', '--extra', $script:Torch, 'uvicorn', 'ppe.api.app:create_app',
                '--factory', '--port', '8000')
        } finally {
            $env:PPE_MODEL__WEIGHTS = $previous
        }
    }
    'clean'        {
        foreach ($p in @('.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov',
                         '.coverage', 'coverage.xml')) {
            if (Test-Path $p) { Remove-Item -Recurse -Force $p }
        }
        Get-ChildItem -Recurse -Directory -Filter '__pycache__' |
            Remove-Item -Recurse -Force
    }
    default {
        Write-Error "Unknown target '$Target'. Run '.\make.ps1 help' for the list."
        exit 1
    }
}
