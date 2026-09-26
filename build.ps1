<#
.SYNOPSIS
  Build the CUDA and CPU matrix-multiplication experiments on Windows.

.DESCRIPTION
  Compiles every .cu file with nvcc and matrix_mul.cpp with the MSVC host
  compiler (cl.exe). nvcc on Windows needs cl.exe; if it is not usable in the
  current shell, the script finds Visual Studio (or Build Tools) with vswhere
  and loads vcvars64.bat into this process first.

.PARAMETER Arch
  GPU architecture passed to nvcc -arch (default sm_86 = RTX 30-series / Ampere).

.PARAMETER OutDir
  Output directory for the executables (default: .\build).

.EXAMPLE
  .\build.ps1
  .\build.ps1 -Arch sm_75 -OutDir C:\temp\mm
#>
param(
    [string]$Arch = "sm_86",
    [string]$OutDir = (Join-Path $PSScriptRoot "build")
)

$ErrorActionPreference = "Stop"

function Import-VcVars {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "cl.exe not usable and vswhere.exe not found. Install Visual Studio Build Tools (C++ workload)." }
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw "No Visual Studio install with the C++ x64 tools was found." }
    $vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
    Write-Host "Loading MSVC environment from $vcvars"
    cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
    }
}

if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "nvcc not found. Install the CUDA Toolkit and make sure its bin directory is on PATH."
}
# cl.exe may be on PATH without INCLUDE/LIB set; vcvars64 sets both.
if (-not (Get-Command cl -ErrorAction SilentlyContinue) -or -not $env:INCLUDE) { Import-VcVars }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

foreach ($name in "hello", "matrix_mul", "matrix_mul_compare") {
    $src = Join-Path $PSScriptRoot "$name.cu"
    $exe = Join-Path $OutDir "$name.exe"
    Write-Host "nvcc  $name.cu -> $exe"
    & nvcc -O2 "-arch=$Arch" -o $exe $src
    if ($LASTEXITCODE -ne 0) { throw "nvcc failed on $name.cu" }
}

# CPU-only baseline, built with the same MSVC compiler.
$cpuExe = Join-Path $OutDir "matrix_mul_cpu.exe"
Write-Host "cl    matrix_mul.cpp -> $cpuExe"
& cl /nologo /O2 /EHsc /std:c++17 "/Fe:$cpuExe" "/Fo:$(Join-Path $OutDir 'matrix_mul_cpu.obj')" (Join-Path $PSScriptRoot "matrix_mul.cpp") | Out-Host
if ($LASTEXITCODE -ne 0) { throw "cl failed on matrix_mul.cpp" }

Write-Host "Done. Executables are in $OutDir"
