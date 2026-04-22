cmake_minimum_required(VERSION 3.22)
project(lemondate LANGUAGES C CXX)

# ============================================================
# lemondate monorepo top-level CMakeLists.
#
# Builds the ROCm/HIP-accelerated voice stack components that sit
# NEXT TO lemond. lemond itself (src/lemond/) is a standalone
# CMake project with its own build tree, driven by build.cmd so we
# don't have to patch its 1400+ line CMakeLists.
#
# Components built here:
#   - src/ggml/             unified ggml tree (llama.cpp base + our
#                           kcpp dirtypatch ops + CUDA kernels for
#                           ttscpp custom ops)
#   - src/ttscpp/           Kokoro TTS inference engine (HIP)
#   - src/whisper/          whisper.cpp + whisper-server.exe (HIP)
#   - src/kokoro-hip-server/ new cpp-httplib server exposing
#                           /v1/audio/speech, powered by ttscpp
#
# External deps (only): TheRock ROCm SDK (hip/rocblas/hipblaslt/
# rocwmma/miopen). Everything else is vendored in src/.
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default the global output dirs so every component drops into
# <build>/bin|lib. build.cmd then stages bin/ into the lemondate
# install tree.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# GFX target(s). Semicolon-separated list, CMake convention.
# Default gfx1201 (Navi 48 / RX 9070 XT). Override:
#   cmake -S . -B build -DGFX_TARGET=gfx1151            (Strix Halo)
#   cmake -S . -B build -DGFX_TARGET=gfx1151\;gfx1201   (fat binary)
if(NOT DEFINED GFX_TARGET)
    set(GFX_TARGET "gfx1201" CACHE STRING "Target AMD GPU arch(s)")
endif()
message(STATUS "lemondate: GFX_TARGET=${GFX_TARGET}")

# Components are added in dependency order. Each is optional at
# scaffold time so `cmake -S . -B build` still works while A2/A3/A7
# are filling in their subdirs in parallel. Once all are present
# the whole stack builds with one invocation.
foreach(component IN ITEMS ggml ttscpp whisper kokoro-hip-server)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/${component}/CMakeLists.txt)
        message(STATUS "lemondate: adding src/${component}")
        add_subdirectory(src/${component})
    else()
        message(STATUS "lemondate: skipping src/${component} (not present yet)")
    endif()
endforeach()
