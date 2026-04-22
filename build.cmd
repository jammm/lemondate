@echo off
REM ============================================================
REM lemondate one-shot build.
REM
REM Builds all four C++ targets from a single invocation:
REM   1. src/ggml + src/ttscpp + src/whisper + src/kokoro-hip-server
REM      (the HIP-accelerated voice stack) via the TOP-LEVEL
REM      CMakeLists.txt. Output binaries go to build\bin\.
REM   2. src/lemond (the lemonade HTTP orchestrator) via its own
REM      CMakeLists.txt at src\lemond\CMakeLists.txt. Output
REM      binaries go to src\lemond\build\bin\.
REM
REM After both succeed we stage every produced .exe/.dll into
REM d:\jam\lemondate\bin\ so the shims in this repo's sister
REM tts_tts_claude_code installer can find them via a single
REM LemondatePath argument.
REM
REM Usage:
REM   build.cmd                     full configure+build
REM   build.cmd configure           configure only (fast smoke test)
REM   build.cmd clean                wipe build dirs and exit
REM
REM Environment overrides:
REM   GFX_TARGET      default gfx1201 (semicolon-separated list
REM                   ok, e.g. gfx1151;gfx1201)
REM   VENV_ROOT       default d:\jam\demos\.venv
REM   BUILD_DIR       default %CD%\build
REM   VS_VCVARS       path to vcvars64.bat (default VS 2022
REM                   Community install)
REM ============================================================

setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
pushd "%~dp0"

if "%GFX_TARGET%"=="" set "GFX_TARGET=gfx1201"
if "%VENV_ROOT%"=="" set "VENV_ROOT=d:\jam\demos\.venv"
if "%BUILD_DIR%"=="" set "BUILD_DIR=%CD%\build"
if "%VS_VCVARS%"=="" set "VS_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

set "ROCM_ROOT=%VENV_ROOT%\Lib\site-packages\_rocm_sdk_devel"
set "ROCM_CLANG=%ROCM_ROOT%\lib\llvm\bin"

REM --- sanity: TheRock ROCm SDK ---
if not exist "%ROCM_CLANG%\clang.exe" (
    echo [lemondate] ERROR: clang.exe not at "%ROCM_CLANG%".
    echo             Run: pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ torch "rocm[libraries,devel]"
    exit /b 1
)

REM --- sanity: VS 2022 x64 env (for rc.exe, Windows SDK headers, link.exe) ---
if not exist "%VS_VCVARS%" (
    echo [lemondate] ERROR: vcvars64.bat not at "%VS_VCVARS%".
    echo             Install Visual Studio 2022 + "Desktop development with C++" workload,
    echo             or set VS_VCVARS to your local vcvars64.bat.
    exit /b 1
)

REM --- clean mode ---
if "%1"=="clean" (
    echo [lemondate] cleaning "%BUILD_DIR%" and "%CD%\src\lemond\build"
    rmdir /s /q "%BUILD_DIR%" 2>nul
    rmdir /s /q "%CD%\src\lemond\build" 2>nul
    echo [lemondate] clean done.
    popd
    endlocal
    exit /b 0
)

echo [lemondate] GFX_TARGET = %GFX_TARGET%
echo [lemondate] VENV_ROOT  = %VENV_ROOT%
echo [lemondate] ROCM_ROOT  = %ROCM_ROOT%
echo [lemondate] BUILD_DIR  = %BUILD_DIR%
echo [lemondate] VS_VCVARS  = %VS_VCVARS%
echo.

REM --- Load VS 2022 x64 environment (brings rc.exe + link.exe + SDK headers) ---
echo [lemondate] loading vcvars64...
call "%VS_VCVARS%" >nul
if errorlevel 1 (
    echo [lemondate] ERROR: vcvars64 failed
    exit /b 1
)

REM --- HIP env the way koboldcpp + whisper builds expect it ---
set "HIP_PATH=%ROCM_ROOT%"
set "HIP_PLATFORM=amd"
set "HIP_CLANG_PATH=%ROCM_CLANG%"
set "CMAKE_PREFIX_PATH=%ROCM_ROOT%\lib\cmake;%ROCM_ROOT%"
set "PATH=%ROCM_ROOT%\bin;%ROCM_CLANG%;%PATH%"

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
    -DCMAKE_PREFIX_PATH="%ROCM_ROOT%\lib\cmake;%ROCM_ROOT%" ^
    -DGGML_HIP=ON ^
    -DGGML_HIP_ROCWMMA_FATTN=OFF ^
    -DGGML_CPU=OFF ^
    -DGGML_BLAS=OFF ^
    -DGGML_METAL=OFF ^
    -DGGML_OPENCL=OFF ^
    -DGGML_VULKAN=OFF ^
    -DGGML_WEBGPU=OFF ^
    -DGGML_HEXAGON=OFF ^
    -DGGML_BACKEND_DL=OFF ^
    -DGFX_TARGET="%GFX_TARGET%" ^
    -DGPU_TARGETS="%GFX_TARGET%" ^
    -DAMDGPU_TARGETS="%GFX_TARGET%" ^
    -DCMAKE_HIP_ARCHITECTURES="%GFX_TARGET%" ^
    -DCMAKE_SYSTEM_VERSION="10.0.26100.0"
if errorlevel 1 goto :fail

if "%1"=="configure" (
    echo.
    echo [lemondate] configure-only mode: skipping build.
    popd
    endlocal
    exit /b 0
)

echo.
echo [lemondate] === Building top-level HIP stack ===
cmake --build "%BUILD_DIR%" --parallel
if errorlevel 1 goto :fail

REM ------------------------------------------------------------
REM Step 2: lemond (lemonade HTTP orchestrator)
REM ------------------------------------------------------------
set "LEMOND_SRC=%CD%\src\lemond"
set "LEMOND_BUILD=%LEMOND_SRC%\build"

echo.
echo [lemondate] === Configuring lemond ===
REM lemond is a plain CPU HTTP server. It does not need HIP/ROCm
REM toolchain. Use real MSVC cl.exe (from vcvars64 above) so the
REM /MT, /Zc:preprocessor, and other MSVC-specific flags that
REM lemonade + libwebsockets/curl use don't trip up clang-cl's
REM stricter handling.
cmake -S "%LEMOND_SRC%" -B "%LEMOND_BUILD%" ^
    -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=cl.exe ^
    -DCMAKE_CXX_COMPILER=cl.exe ^
    -DCMAKE_SYSTEM_VERSION="10.0.26100.0"
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

REM copy /y silently succeeds-with-exit-1 when the destination is locked
REM (another process is running it), which used to leave a stale binary in
REM bin\ while the build log claimed success. Check the exit code and abort
REM loudly so the user knows to stop services before re-staging.
set _stage_failed=0
if exist "%BUILD_DIR%\bin\whisper-server.exe" (
    copy /y "%BUILD_DIR%\bin\whisper-server.exe" "%CD%\bin\" >nul
    if errorlevel 1 set _stage_failed=1
)
if exist "%BUILD_DIR%\bin\kokoro-hip-server.exe" (
    copy /y "%BUILD_DIR%\bin\kokoro-hip-server.exe" "%CD%\bin\" >nul
    if errorlevel 1 set _stage_failed=1
)
for %%F in ("%BUILD_DIR%\bin\*.dll") do copy /y "%%F" "%CD%\bin\" >nul 2>&1
if "%_stage_failed%"=="1" (
    echo.
    echo [lemondate] ERROR: staging to bin\ failed - a target binary is
    echo             likely locked by a running process. Stop services
    echo             (tts_tts_claude_code\installers\stop_services.ps1)
    echo             then re-run build.cmd.
    goto :fail
)

REM lemond's CMake outputs: lemond.exe at build root, other exes at build\Release\
if exist "%LEMOND_BUILD%\lemond.exe"                copy /y "%LEMOND_BUILD%\lemond.exe"                "%CD%\bin\" >nul
if exist "%LEMOND_BUILD%\Release\lemonade.exe"      copy /y "%LEMOND_BUILD%\Release\lemonade.exe"      "%CD%\bin\" >nul
if exist "%LEMOND_BUILD%\Release\LemonadeServer.exe" copy /y "%LEMOND_BUILD%\Release\LemonadeServer.exe" "%CD%\bin\" >nul
for %%F in ("%LEMOND_BUILD%\*.dll") do copy /y "%%F" "%CD%\bin\" >nul 2>&1

REM Stage lemond's resources (defaults.json, server_models.json, etc.) next
REM to lemond.exe. ConfigFile::get_defaults() looks them up via
REM get_resource_path("resources/..."), which searches relative to the
REM executable's dir first.
if not exist "%CD%\bin\resources" mkdir "%CD%\bin\resources"
xcopy /E /Y /Q /I "%LEMOND_SRC%\src\cpp\resources" "%CD%\bin\resources" >nul

REM lemond's libcurl links against zlib1 + libssh2 as DLLs. On this
REM build host those resolve to Strawberry Perl's C toolchain; copy
REM them into bin so end-users don't need Strawberry Perl on PATH.
REM Fall through silently if the install path differs.
if exist "C:\Strawberry\c\bin\zlib1__.dll"     copy /y "C:\Strawberry\c\bin\zlib1__.dll"     "%CD%\bin\" >nul 2>&1
if exist "C:\Strawberry\c\bin\libssh2-1__.dll" copy /y "C:\Strawberry\c\bin\libssh2-1__.dll" "%CD%\bin\" >nul 2>&1

REM rocFFT runtime (needed by ggml-hip for the ttscpp STFT/ISTFT path)
if exist "%ROCM_ROOT%\bin\rocfft.dll" copy /y "%ROCM_ROOT%\bin\rocfft.dll" "%CD%\bin\" >nul 2>&1

REM Kokoro IPA override dictionary — 65k entries that get hit BEFORE the
REM rule-based phonemizer's buggy cascade, fixing most `-es` plural /
REM `-s` 3sg / common-word mispronunciations for free. kokoro-hip-server
REM reads this at startup via populate_kokoro_ipa_map.
if exist "%CD%\assets\embd_res\kokoro_ipa.embd" (
    if not exist "%CD%\bin\embd_res" mkdir "%CD%\bin\embd_res"
    copy /y "%CD%\assets\embd_res\kokoro_ipa.embd" "%CD%\bin\embd_res\" >nul 2>&1
)

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
