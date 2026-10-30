@echo off
rem =============================================================================
rem kernel-ops | tools/build-windows.bat
rem Configure + build with Ninja and ccache on Windows (MSVC + CUDA).
rem
rem   tools\build-windows.bat                 -> Release, build-ninja\
rem   tools\build-windows.bat Debug           -> Debug (slow; symbols)
rem   tools\build-windows.bat Release out\b   -> custom build directory
rem
rem Why Ninja: CMake only honours CMAKE_{CXX,CUDA}_COMPILER_LAUNCHER with the
rem Makefile/Ninja generators. The Visual Studio generator silently ignores
rem ccache, so the kernel rebuild after a header change stays expensive.
rem
rem Requirements: Visual Studio 2022 (C++ tools), CUDA toolkit, ninja on PATH or
rem tools\ninja\ninja.exe, ccache on PATH or KOPS_CCACHE_PROGRAM set to its path.
rem =============================================================================
setlocal EnableDelayedExpansion

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "CFG=%~1"
if "%CFG%"=="" set "CFG=Release"
set "BUILD=%~2"
if "%BUILD%"=="" set "BUILD=%ROOT%\build-ninja"
if not defined KOPS_CUDA_ARCHS set "KOPS_CUDA_ARCHS=native"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "!VSWHERE!" (
  echo [error] vswhere not found: install Visual Studio 2022 with the C++ workload.
  exit /b 1
)
for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH (
  echo [error] Visual Studio installation not found.
  exit /b 1
)
call "!VSPATH!\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
  echo [error] vcvars64.bat failed.
  exit /b 1
)

set "NINJA="
if defined KOPS_NINJA set "NINJA=!KOPS_NINJA!"
if not defined NINJA where ninja >nul 2>nul && set "NINJA=ninja"
if not defined NINJA if exist "%ROOT%\tools\ninja\ninja.exe" set "NINJA=%ROOT%\tools\ninja\ninja.exe"
if not defined NINJA (
  echo [error] ninja not found: put it on PATH or at tools\ninja\ninja.exe
  exit /b 1
)

if not defined KOPS_CCACHE_PROGRAM (
  where ccache >nul 2>nul && for /f "delims=" %%i in ('where ccache') do set "KOPS_CCACHE_PROGRAM=%%i"
)
if defined KOPS_CCACHE_PROGRAM (
  set "CCACHE_SLOPPINESS=time_macros,random_seed"
  if not defined CCACHE_MAXSIZE set "CCACHE_MAXSIZE=5G"
  echo [info] ccache = !KOPS_CCACHE_PROGRAM!
) else (
  echo [warn] ccache not found: build will not be cached.
)

echo [info] configure %CFG% -> %BUILD%
cmake -S "%ROOT%" -B "%BUILD%" -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="!NINJA!" ^
  -DKOPS_BUILD_TYPE=%CFG% ^
  -DKOPS_CUDA_ARCHS=%KOPS_CUDA_ARCHS% ^
  -DKOPS_CCACHE_PROGRAM="!KOPS_CCACHE_PROGRAM!"
if errorlevel 1 exit /b 1

echo [info] build
cmake --build "%BUILD%" -j %NUMBER_OF_PROCESSORS%
if errorlevel 1 exit /b 1

echo [info] test
ctest --test-dir "%BUILD%" --output-on-failure
exit /b %errorlevel%
