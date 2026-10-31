@echo off
rem =============================================================================
rem kernel-ops | tools/build-windows.bat
rem Configure + build + test through the CMake presets (Ninja + ccache).
rem
rem   tools\build-windows.bat              -> dev preset (native arch, Release)
rem   tools\build-windows.bat dev-debug    -> RelWithDebInfo
rem   tools\build-windows.bat release      -> multi-arch, no tests
rem
rem Requirements: Visual Studio 2022 (C++ tools), CUDA toolkit, ninja, ccache.
rem Why Ninja: CMake ignores CMAKE_*_COMPILER_LAUNCHER with the Visual Studio
rem generator, so a VS build silently loses the compiler cache.
rem =============================================================================
setlocal EnableDelayedExpansion

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PRESET=%~1"
if "%PRESET%"=="" set "PRESET=dev"

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

rem Ninja is required for the compiler launcher. tools\ninja\ninja.exe is used
rem when ninja is not on PATH.
where ninja >nul 2>nul
if errorlevel 1 (
  if exist "%ROOT%\tools\ninja\ninja.exe" (
    set "PATH=%ROOT%\tools\ninja;!PATH!"
    echo [info] ninja = %ROOT%\tools\ninja\ninja.exe
  ) else (
    echo [error] ninja not found.
    echo         install it with:  winget install Ninja-build.Ninja
    echo         or:               python -m pip install ninja
    echo         or place ninja.exe at tools\ninja\ninja.exe
    exit /b 1
  )
)

if defined KOPS_CCACHE_PROGRAM (
  echo [info] ccache = !KOPS_CCACHE_PROGRAM!
) else (
  where ccache >nul 2>nul
  if errorlevel 1 (
    if exist "%ROOT%\tools\ccache\ccache.exe" (
      set "PATH=%ROOT%\tools\ccache;!PATH!"
      echo [info] ccache = %ROOT%\tools\ccache\ccache.exe
    ) else (
      echo [warn] ccache not found: builds work but are not cached.
      echo        install it with:  winget install ccache.ccache
      echo        or place ccache.exe at tools\ccache\ccache.exe
    )
  )
)

cd /d "%ROOT%"
echo [info] configure preset %PRESET%
cmake --preset %PRESET%
if errorlevel 1 exit /b 1

echo [info] build preset %PRESET%
cmake --build --preset %PRESET%
if errorlevel 1 exit /b 1

if exist "%ROOT%\build\%PRESET%\CTestTestfile.cmake" (
  echo [info] test preset %PRESET%
  ctest --test-dir "%ROOT%\build\%PRESET%" --output-on-failure
  exit /b %errorlevel%
)
exit /b 0
