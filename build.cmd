@echo off
REM ============================================================
REM lemondate one-shot build.
REM
REM Builds all four C++ targets from a single invocation:
REM   1. src/ggml + src/ttscpp + src/whisper + src/kokoro-hip-server
REM      (the HIP-accelerated voice stack) via the TOP-LEVEL
REM      CMakeLists at d:\jam\lemondate\CMakeLists.txt. Output
REM      binaries go to build\bin\.
REM   2. src/lemond (the lemonade HTTP orchestrator) via its own
REM      CMakeLists at src\lemond\CMakeLists.txt. Output binaries
REM      go to src\lemond\build\bin\.
REM
REM After both succeed we stage every produced .exe/.dll into
REM d:\jam\lemondate\bin\ so the shims in this repo's sister
REM tts_tts_claude_code installer can find them via a single
REM LemondatePath argument.
REM
REM Prereqs:
REM   - VS 2022 x64 Developer PowerShell / Command Prompt on PATH
REM   - TheRock ROCm SDK installed into the python venv at %VENV%
REM     (default d:\jam\demos\.venv); CMake picks up its clang
REM     toolchain automatically.
REM
REM Environment overrides:
REM   GFX_TARGET      default gfx1201 (semicolon-separated list
REM                   ok, e.g. gfx1151;gfx1201)
REM   VENV_ROOT       default d:\jam\demos\.venv
REM   BUILD_DIR       default %CD%\build
REM ============================================================

setlocal ENABLEEXTENSIONS
pushd "%~dp0"

if "%GFX_TARGET%"=="" set GFX_TARGET=gfx1201
if "%VENV_ROOT%"=="" set VENV_ROOT=d:\jam\demos\.venv
if "%BUILD_DIR%"=="" set BUILD_DIR=%CD%\build

set ROCM_ROOT=%VENV_ROOT%\Lib\site-packages\_rocm_sdk_devel
set ROCM_CLANG=%ROCM_ROOT%\lib\llvm\bin

if not exist "%ROCM_CLANG%\clang.exe" (
    echo [lemondate] ERROR: could not find clang.exe at "%ROCM_CLANG%".
    echo             Install TheRock ROCm SDK into the venv first:
    echo               pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ torch "rocm[libraries,devel]"
    exit /b 1
)

echo [lemondate] GFX_TARGET = %GFX_TARGET%
echo [lemondate] VENV_ROOT  = %VENV_ROOT%
echo [lemondate] BUILD_DIR  = %BUILD_DIR%

REM ------------------------------------------------------------
REM Step 1: top-level HIP stack (ggml / ttscpp / whisper / kokoro-hip-server)
REM ------------------------------------------------------------
echo.
echo [lemondate] === Configuring top-level HIP stack ===
cmake -S . -B "%BUILD_DIR%" ^
    -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER="%ROCM_CLANG%\clang.exe" ^
    -DCMAKE_CXX_COMPILER="%ROCM_CLANG%\clang++.exe" ^
    -DGFX_TARGET="%GFX_TARGET%" ^
    -DCMAKE_SYSTEM_VERSION="10.0.26100.0"
if errorlevel 1 goto :fail

echo.
echo [lemondate] === Building top-level HIP stack ===
cmake --build "%BUILD_DIR%" --parallel
if errorlevel 1 goto :fail

REM ------------------------------------------------------------
REM Step 2: lemond (lemonade HTTP orchestrator)
REM ------------------------------------------------------------
set LEMOND_SRC=%CD%\src\lemond
set LEMOND_BUILD=%LEMOND_SRC%\build

echo.
echo [lemondate] === Configuring lemond ===
cmake -S "%LEMOND_SRC%" -B "%LEMOND_BUILD%" ^
    -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 goto :fail

echo.
echo [lemondate] === Building lemond ===
cmake --build "%LEMOND_BUILD%" --parallel
if errorlevel 1 goto :fail

REM ------------------------------------------------------------
REM Step 3: stage bin/
REM ------------------------------------------------------------
echo.
echo [lemondate] === Staging bin\ ===
if not exist "%CD%\bin" mkdir "%CD%\bin"

REM Top-level artefacts
if exist "%BUILD_DIR%\bin\whisper-server.exe" (
    copy /y "%BUILD_DIR%\bin\whisper-server.exe" "%CD%\bin\" >nul
)
if exist "%BUILD_DIR%\bin\kokoro-hip-server.exe" (
    copy /y "%BUILD_DIR%\bin\kokoro-hip-server.exe" "%CD%\bin\" >nul
)
for %%F in ("%BUILD_DIR%\bin\*.dll") do copy /y "%%F" "%CD%\bin\" >nul 2>&1

REM lemond artefacts
if exist "%LEMOND_BUILD%\bin\lemond.exe" (
    copy /y "%LEMOND_BUILD%\bin\lemond.exe" "%CD%\bin\" >nul
)
if exist "%LEMOND_BUILD%\bin\lemonade.exe" (
    copy /y "%LEMOND_BUILD%\bin\lemonade.exe" "%CD%\bin\" >nul
)
for %%F in ("%LEMOND_BUILD%\bin\*.dll") do copy /y "%%F" "%CD%\bin\" >nul 2>&1

echo.
echo [lemondate] Build complete. Binaries staged under "%CD%\bin\".
popd
endlocal
exit /b 0

:fail
echo [lemondate] BUILD FAILED.
popd
endlocal
exit /b 1
