#pragma once

#include "common.cuh"

// CUDA implementations of the ttscpp custom ops that ggml.c declares
// as kcpp dirtypatches (ggml.h GGML_OP_RECIPROCAL / GGML_OP_TTSROUND /
// GGML_OP_MOD / GGML_OP_CUMSUM_TTS). Without these, the scheduler
// dispatches every call to CPU and inserts CPU<->GPU memcpy + sync
// boundaries, which dominates Kokoro's per-token cost on the HIP path
// (the noise/sin generator runs ggml_mod once per generator block and
// ggml_cumsum_tts once per build_sin_gen call).

void ggml_cuda_op_reciprocal(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_ttsround  (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mod       (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_cumsum_tts(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
