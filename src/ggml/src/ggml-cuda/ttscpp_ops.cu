// CUDA implementations for the kcpp ttscpp custom ops. See header for
// rationale. All four are simple element-wise / row-stride kernels —
// the win isn't kernel speed (any of these would run faster on CPU
// for a tiny tensor), it's avoiding the GPU->host->GPU bounce around
// each call when the surrounding ops live on the GPU backend.

#include "ttscpp_ops.cuh"

#include <cmath>

#define TTSCPP_BLOCK 256

template <typename T>
static __global__ void k_reciprocal(const T * __restrict__ x, T * __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = (T) (1.0f / (float) x[i]);
    }
}

template <typename T>
static __global__ void k_ttsround(const T * __restrict__ x, T * __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Match the CPU impl in ggml-cpu.c: (float)((int)(x + 0.5f)).
        // Truncating cast == round-half-up for positive, round-half-down
        // for negative. We replicate that exactly so behaviour matches
        // CPU for graphs that mix backends.
        const float v = (float) x[i];
        y[i] = (T) (float)((int) (v + 0.5f));
    }
}

template <typename T>
static __global__ void k_mod(const T * __restrict__ x, T * __restrict__ y, int64_t n, float mod_val) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = (T) fmodf((float) x[i], mod_val);
    }
}

// One block handles one (row, batch) cumulative sum along ne0. Uses a
// single thread per row (cumsum is inherently sequential) but launches
// many rows in parallel so a (256, 64) tensor uses 64 blocks. This is
// the same threading granularity as the CPU implementation in
// ggml-cpu.c (ggml_compute_forward_cumsum_f32_tts) — Kokoro only ever
// calls cumsum_tts on a (length, harmonic_num) shape with length up
// to a few thousand, so a smarter parallel-prefix scan isn't worth
// the extra complexity yet.
static __global__ void k_cumsum_tts_f32(const float * __restrict__ src, float * __restrict__ dst,
        int64_t ne0, int64_t ne1, int64_t ne2,
        int64_t s_src1, int64_t s_src2,
        int64_t s_dst1, int64_t s_dst2) {
    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.y;
    if (i1 >= ne1 || i2 >= ne2) {
        return;
    }
    const float * srow = src + i2 * s_src2 + i1 * s_src1;
    float * drow = dst + i2 * s_dst2 + i1 * s_dst1;
    float acc = 0.0f;
    for (int64_t i0 = 0; i0 < ne0; i0++) {
        acc += srow[i0];
        drow[i0] = acc;
    }
}

void ggml_cuda_op_reciprocal(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));
    const int64_t n = ggml_nelements(src);
    const int64_t blocks = (n + TTSCPP_BLOCK - 1) / TTSCPP_BLOCK;
    k_reciprocal<float><<<blocks, TTSCPP_BLOCK, 0, ctx.stream()>>>(
        (const float *) src->data, (float *) dst->data, n);
}

void ggml_cuda_op_ttsround(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));
    const int64_t n = ggml_nelements(src);
    const int64_t blocks = (n + TTSCPP_BLOCK - 1) / TTSCPP_BLOCK;
    k_ttsround<float><<<blocks, TTSCPP_BLOCK, 0, ctx.stream()>>>(
        (const float *) src->data, (float *) dst->data, n);
}

void ggml_cuda_op_mod(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));
    const int64_t n = ggml_nelements(src);
    const float mod_val = ((const float *) dst->op_params)[0];
    const int64_t blocks = (n + TTSCPP_BLOCK - 1) / TTSCPP_BLOCK;
    k_mod<float><<<blocks, TTSCPP_BLOCK, 0, ctx.stream()>>>(
        (const float *) src->data, (float *) dst->data, n, mod_val);
}

void ggml_cuda_op_cumsum_tts(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[0] == sizeof(float));

    const int64_t ne0 = src->ne[0];
    const int64_t ne1 = src->ne[1];
    const int64_t ne2 = src->ne[2];

    const int64_t s_src1 = src->nb[1] / sizeof(float);
    const int64_t s_src2 = src->nb[2] / sizeof(float);
    const int64_t s_dst1 = dst->nb[1] / sizeof(float);
    const int64_t s_dst2 = dst->nb[2] / sizeof(float);

    dim3 grid((unsigned int) ne1, (unsigned int) MAX(ne2, 1), 1);
    k_cumsum_tts_f32<<<grid, 1, 0, ctx.stream()>>>(
        (const float *) src->data, (float *) dst->data,
        ne0, ne1, MAX(ne2, (int64_t)1),
        s_src1, s_src2, s_dst1, s_dst2);
}
