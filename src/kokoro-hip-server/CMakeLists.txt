# ============================================================
# src/kokoro-hip-server/CMakeLists.txt
#
# Builds kokoro-hip-server.exe — a small cpp-httplib server
# that wraps ttscpp's Kokoro inference and exposes an
# OpenAI-compatible /v1/audio/speech endpoint.
#
# Depends on:
#   - ttscpp  (src/ttscpp/, provides ttscpp.h + libttscpp.a)
#   - ggml    (src/ggml/,   linked transitively via ttscpp)
#   - cpp-httplib + nlohmann_json (system first, FetchContent
#     fallback). Versions mirror src/lemond/CMakeLists.txt so
#     both halves of lemondate stay on the same pins.
#
# lemond itself is a standalone CMake project driven by
# build.cmd, so it doesn't participate in this top-level build;
# we fetch our own copy of the deps here rather than reach
# into src/lemond/.
# ============================================================

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Dependency version floor — keep in lockstep with
# src/lemond/CMakeLists.txt (MIN_NLOHMANN_JSON_VERSION,
# MIN_HTTPLIB_VERSION there).
set(KHS_MIN_NLOHMANN_JSON_VERSION "3.11.3")
set(KHS_MIN_HTTPLIB_VERSION       "0.26.0")

find_package(PkgConfig QUIET)

# --- nlohmann_json ------------------------------------------
find_package(nlohmann_json ${KHS_MIN_NLOHMANN_JSON_VERSION} QUIET)
if(nlohmann_json_FOUND)
    message(STATUS "kokoro-hip-server: using system nlohmann_json (>= ${KHS_MIN_NLOHMANN_JSON_VERSION})")
else()
    message(STATUS "kokoro-hip-server: system nlohmann_json not found, will FetchContent")
    include(FetchContent)
    FetchContent_Declare(khs_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v${KHS_MIN_NLOHMANN_JSON_VERSION}
        GIT_SHALLOW TRUE
    )
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    set(JSON_Install    OFF CACHE INTERNAL "")
    FetchContent_MakeAvailable(khs_json)
endif()

# --- cpp-httplib --------------------------------------------
set(KHS_HTTPLIB_SYSTEM OFF)
if(PkgConfig_FOUND)
    pkg_check_modules(KHS_HTTPLIB QUIET cpp-httplib>=${KHS_MIN_HTTPLIB_VERSION})
    if(KHS_HTTPLIB_FOUND)
        set(KHS_HTTPLIB_SYSTEM ON)
    endif()
endif()
if(NOT KHS_HTTPLIB_SYSTEM)
    find_path(KHS_HTTPLIB_INCLUDE_DIRS "httplib.h")
    if(KHS_HTTPLIB_INCLUDE_DIRS)
        set(KHS_HTTPLIB_SYSTEM ON)
    endif()
endif()

if(KHS_HTTPLIB_SYSTEM)
    message(STATUS "kokoro-hip-server: using system cpp-httplib (>= ${KHS_MIN_HTTPLIB_VERSION})")
else()
    message(STATUS "kokoro-hip-server: system cpp-httplib not found, will FetchContent")
    include(FetchContent)
    FetchContent_Declare(khs_httplib
        GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
        GIT_TAG v${KHS_MIN_HTTPLIB_VERSION}
        GIT_SHALLOW TRUE
    )
    # httplib is heavy if all optional features are on; we don't
    # need zstd/openssl for a localhost TTS endpoint, so turn
    # them off to avoid dragging in extra deps.
    set(HTTPLIB_REQUIRE_OPENSSL       OFF CACHE INTERNAL "")
    set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE INTERNAL "")
    set(HTTPLIB_REQUIRE_ZSTD          OFF CACHE INTERNAL "")
    set(HTTPLIB_USE_ZSTD_IF_AVAILABLE OFF CACHE INTERNAL "")
    set(HTTPLIB_INSTALL               OFF CACHE INTERNAL "")
    FetchContent_MakeAvailable(khs_httplib)
endif()

# --- the exe ------------------------------------------------
add_executable(kokoro-hip-server main.cpp)

# Link ttscpp. That target transitively exports ggml + its own
# include dirs, so we don't need to mention either explicitly.
if(TARGET ttscpp)
    target_link_libraries(kokoro-hip-server PRIVATE ttscpp)
else()
    message(FATAL_ERROR "kokoro-hip-server: ttscpp target not found. "
                        "src/ttscpp/CMakeLists.txt must be added before "
                        "src/kokoro-hip-server/CMakeLists.txt.")
endif()
# Belt-and-braces: explicitly link ggml too in case ttscpp only
# exports it PRIVATE for some build configuration.
if(TARGET ggml)
    target_link_libraries(kokoro-hip-server PRIVATE ggml)
endif()

# JSON
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(kokoro-hip-server PRIVATE nlohmann_json::nlohmann_json)
endif()

# httplib
if(KHS_HTTPLIB_SYSTEM)
    if(KHS_HTTPLIB_INCLUDE_DIRS)
        target_include_directories(kokoro-hip-server PRIVATE ${KHS_HTTPLIB_INCLUDE_DIRS})
    endif()
    if(KHS_HTTPLIB_LIBRARIES)
        target_link_libraries(kokoro-hip-server PRIVATE ${KHS_HTTPLIB_LIBRARIES})
    endif()
    if(KHS_HTTPLIB_LIBRARY_DIRS)
        target_link_directories(kokoro-hip-server PRIVATE ${KHS_HTTPLIB_LIBRARY_DIRS})
    endif()
    if(KHS_HTTPLIB_CFLAGS_OTHER)
        target_compile_options(kokoro-hip-server PRIVATE ${KHS_HTTPLIB_CFLAGS_OTHER})
    endif()
else()
    if(TARGET httplib::httplib)
        target_link_libraries(kokoro-hip-server PRIVATE httplib::httplib)
    elseif(TARGET httplib)
        target_link_libraries(kokoro-hip-server PRIVATE httplib)
    endif()
endif()

target_compile_features(kokoro-hip-server PRIVATE cxx_std_17)
set_target_properties(kokoro-hip-server PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(kokoro-hip-server PRIVATE /utf-8)
endif()

# Windows needs ws2_32 / winsock linked for httplib::Server.
# httplib's target normally handles this, but be defensive when
# using a system header-only install.
if(WIN32)
    target_link_libraries(kokoro-hip-server PRIVATE ws2_32)
endif()

message(STATUS "kokoro-hip-server: executable configured")
