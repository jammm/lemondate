// Fused snake activation kernel: dst = a + sin^2(a * alpha) / alpha
//
// alpha is broadcast across ne0 and (optionally) ne2/ne3 — see the
// CPU implementation in ggml-cpu.c::ggml_compute_forward_snake_1d_f32
// for the reference. Calling sequence in Kokoro replaces a 7-op
// subgraph (mul, sin, sqr, mul, div, mul, add — see ttsutil.cpp's
// snake_1d) with this one kernel, eliminating six kernel launches and
// six intermediate tensor allocations per call. Snake_1d is invoked
// many times per inference (every AdaIN res block in the decoder +
// every noise res block in the generator), so this is the single
// largest source of per-op launch overhead in Kokoro on HIP.

#include "snake.cuh"

#include <cmath>

#define SNAKE_BLOCK 256

template <typename T>
static __global__ void k_snake_1d(
        const T * __restrict__ a,
        const T * __restrict__ alpha,
        T * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int64_t s00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s11, const int64_t s12, const int64_t s13,
        const int64_t d0,  const int64_t d1,  const int64_t d2,  const int64_t d3,
        const bool bcast2, const bool bcast3) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = ne0 * ne1 * ne2 * ne3;
    if (i >= total) {
        return;
    }

    const int64_t i0 =  i % ne0;
    const int64_t i1 = (i / ne0) % ne1;
    const int64_t i2 = (i / (ne0 * ne1)) % ne2;
    const int64_t i3 =  i / (ne0 * ne1 * ne2);

    const int64_t j2 = bcast2 ? 0 : i2;
    const int64_t j3 = bcast3 ? 0 : i3;

    const T av = a[i0 * s00 + i1 * s01 + i2 * s02 + i3 * s03];
    const T al = alpha[i1 * s11 + j2 * s12 + j3 * s13];

    const float af = (float) av;
    const float lf = (float) al;
    const float sx = sinf(af * lf);
    const float r  = af + sx * sx / lf;

    dst[i0 * d0 + i1 * d1 + i2 * d2 + i3 * d3] = (T) r;
}

void ggml_cuda_op_snake_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a     = dst->src[0];
    const ggml_tensor * alpha = dst->src[1];

    GGML_ASSERT(a->type     == GGML_TYPE_F32);
    GGML_ASSERT(alpha->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);

    GGML_ASSERT(alpha->ne[0] == 1);
    GGML_ASSERT(alpha->ne[1] == a->ne[1]);
    GGML_ASSERT(alpha->ne[2] == 1 || alpha->ne[2] == a->ne[2]);
    GGML_ASSERT(alpha->ne[3] == 1 || alpha->ne[3] == a->ne[3]);

    const int64_t ne0 = a->ne[0];
    const int64_t ne1 = a->ne[1];
    const int64_t ne2 = a->ne[2];
    const int64_t ne3 = a->ne[3];

    // Stride-in-elements (kernel works in element units, not bytes).
    const size_t es = sizeof(float);
    const int64_t s00 = a->nb[0] / es;
    const int64_t s01 = a->nb[1] / es;
    const int64_t s02 = a->nb[2] / es;
    const int64_t s03 = a->nb[3] / es;

    // alpha has ne[0]==1 so its s10 contribution is zeroed out.
    const int64_t s11 = alpha->nb[1] / es;
    const int64_t s12 = alpha->nb[2] / es;
    const int64_t s13 = alpha->nb[3] / es;

    const int64_t d0 = dst->nb[0] / es;
    const int64_t d1 = dst->nb[1] / es;
    const int64_t d2 = dst->nb[2] / es;
    const int64_t d3 = dst->nb[3] / es;

    const bool bcast2 = alpha->ne[2] == 1 && a->ne[2] != 1;
    const bool bcast3 = alpha->ne[3] == 1 && a->ne[3] != 1;

    const int64_t total = ne0 * ne1 * ne2 * ne3;
    const int64_t blocks = (total + SNAKE_BLOCK - 1) / SNAKE_BLOCK;

    cudaStream_t stream = ctx.stream();
    k_snake_1d<float><<<blocks, SNAKE_BLOCK, 0, stream>>>(
        (const float *) a->data, (const float *) alpha->data, (float *) dst->data,
        ne0, ne1, ne2, ne3,
        s00, s01, s02, s03,
        s11, s12, s13,
        d0, d1, d2, d3,
        bcast2, bcast3);
}
