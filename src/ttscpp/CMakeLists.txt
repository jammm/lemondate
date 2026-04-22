# ============================================================
# src/ttscpp/CMakeLists.txt
#
# Builds the ttscpp static library — Kokoro TTS inference on
# top of ggml. Lemondate ships Kokoro as the only TTS model,
# so the Parler / Dia / Orpheus / (t5/dac/snac/general_neural_
# audio_codec) sources that were originally vendored alongside
# Kokoro have been deleted. The remaining sources assume a
# GPU-only ggml (GGML_CPU=OFF) and never call into the CPU
# compute backend.
#
# Source was vendored from koboldcpp/otherarch/ttscpp/ (see
# README / plan for the commit it was pulled from). Upstream
# has no CMakeLists of its own — koboldcpp builds it via a
# unity-include pattern in otherarch/tts_adapter.cpp. We build
# each .cpp as a real translation unit here so that lemond's
# kokoro-hip-server.exe can link the library cleanly.
#
# The target is called `ttscpp` and links PUBLIC to `ggml`
# (provided by src/ggml/CMakeLists.txt, which is added before
# this subdir by the top-level foreach).
# ============================================================

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

set(TTSCPP_SOURCES
    src/kokoro_model.cpp
    src/phonemizer.cpp
    src/ttsargs.cpp
    src/ttscpp.cpp
    src/ttssampler.cpp
    src/ttstokenizer.cpp
    src/ttsutil.cpp
    src/tts_model.cpp
)

add_library(ttscpp STATIC ${TTSCPP_SOURCES})

# ttscpp.h (in include/) transitively #includes kokoro_model.h,
# parler_model.h, etc. which live in src/. Consumers therefore
# need BOTH directories on their include path, so we expose
# them PUBLIC. This mirrors what koboldcpp's tts_adapter target
# did upstream.
target_include_directories(ttscpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link the unified ggml target produced by src/ggml/. The
# top-level CMake adds src/ggml before src/ttscpp so the
# target is already defined here. PUBLIC so that consumers
# (kokoro-hip-server) pick up ggml headers transitively.
if(TARGET ggml)
    target_link_libraries(ttscpp PUBLIC ggml)
else()
    message(WARNING "src/ttscpp: ggml target not found; ttscpp will not link. "
                    "Make sure src/ggml/CMakeLists.txt defines an imported or "
                    "real target named `ggml` and is added before src/ttscpp.")
endif()

# ttscpp's ttsutil.h unconditionally #includes "ggml-impl.h",
# which lives in ggml's src/ tree (INTERNAL to ggml). llama.cpp's
# ggml target only exposes include/ as PUBLIC, not src/. Upstream
# koboldcpp worked around this by adding ./ggml/src to
# tts_adapter's include dirs. Mirror that here so ttscpp builds
# cleanly against a vanilla llama.cpp ggml tree. PUBLIC because
# kokoro-hip-server (consumer of ttscpp.h) pulls ttsutil.h
# transitively too.
set(_ttscpp_ggml_src "${CMAKE_SOURCE_DIR}/src/ggml/src")
if(EXISTS "${_ttscpp_ggml_src}/ggml-impl.h")
    target_include_directories(ttscpp PUBLIC ${_ttscpp_ggml_src})
else()
    message(WARNING "src/ttscpp: ggml-impl.h not found at ${_ttscpp_ggml_src}; "
                    "ttsutil.h may fail to compile. Verify src/ggml/ tree.")
endif()

target_compile_features(ttscpp PUBLIC cxx_std_17)
set_target_properties(ttscpp PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ttscpp has UTF-8 literals (phonemizer tables, Kokoro voice
# names). MSVC needs /utf-8 or it mangles them. No-op on clang/gcc.
if(MSVC)
    target_compile_options(ttscpp PRIVATE /utf-8)
endif()

# Silence a handful of warnings that are noise in a vendored tree.
if(MSVC)
    target_compile_options(ttscpp PRIVATE /wd4244 /wd4267 /wd4996)
else()
    target_compile_options(ttscpp PRIVATE
        -Wno-unused-variable
        -Wno-unused-function
        -Wno-sign-compare
    )
endif()

message(STATUS "ttscpp: static library configured (${CMAKE_CURRENT_SOURCE_DIR})")
