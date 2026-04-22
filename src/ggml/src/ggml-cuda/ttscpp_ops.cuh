#pragma once

#include "common.cuh"

// CUDA implementations of the ttscpp custom ops that ggml.c declares
// as kcpp dirtypatches (ggml.h GGML_OP_RECIPROCAL / GGML_OP_TTSROUND /
// GGML_OP_MOD / GGML_OP_CUMSUM_TTS / GGML_OP_STFT / GGML_OP_AA_STFT /
// GGML_OP_ISTFT / GGML_OP_AA_ISTFT / GGML_OP_CONV_TRANSPOSE_1D_TTS /
// GGML_OP_UPSCALE_LINEAR / GGML_OP_UV_NOISE). Without these, the
// scheduler dispatches every call to CPU and inserts CPU<->GPU memcpy +
// sync boundaries, which dominates Kokoro's per-token cost on the HIP
// path (the vocoder alone hits STFT/ISTFT/CONV_TRANSPOSE_1D_TTS/
// UPSCALE_LINEAR several times per utterance).

void ggml_cuda_op_reciprocal          (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_ttsround            (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mod                 (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_cumsum_tts          (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_uv_noise            (ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_stft                (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_aa_stft             (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_istft               (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_aa_istft            (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_conv_transpose_1d_tts(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_upscale_linear      (ggml_backend_cuda_context & ctx, ggml_tensor * dst);
