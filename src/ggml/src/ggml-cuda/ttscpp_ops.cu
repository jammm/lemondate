// CUDA implementations for the kcpp ttscpp custom ops. See header for
// rationale. Originally the file only covered the element-wise and
// row-stride ops (reciprocal, ttsround, mod, cumsum_tts, uv_noise);
// it has since grown to cover the vocoder-shaped ops too (STFT /
// AA_STFT / ISTFT / AA_ISTFT / CONV_TRANSPOSE_1D_TTS / UPSCALE_LINEAR).
// The original scalar kernels are kept together near the top, the
// vocoder kernels follow below.
//
// The core win isn't kernel speed for the scalar ops (any of these
// would run faster on CPU for a tiny tensor) — it's avoiding the
// GPU->host->GPU bounce around each call when the surrounding ops
// live on the GPU backend. For the larger vocoder ops (STFT/ISTFT/
// transposed conv/upsample), the GPU is plainly a win.

#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

// Fused Kokoro voice-gate + noise kernel. Replaces the ggml_map_custom3
// call in build_sin_gen (kokoro_model.cpp) that used to force the
// surrounding generator subgraph to the CPU backend and cost one
// GPU->host->GPU bounce per token. See uv_noise_compute() in
// ttscpp/src/ttsutil.cpp for the reference implementation — this is
// the same scalar logic, parallelised one thread per sequence
// position.
//
// Layout:
//   dst    : F32 [sequence_length, harmonic_num, 2], contiguous.
//            plane 0 (elements [0, S*H)) is the uv gate,
//            plane 1 (elements [S*H, 2*S*H)) is the scaled noise.
//   gate   : F32 [sequence_length], contiguous — upscaled f0 curve.
//   params : first 4 elements are the scalar thresholds
//            (voice_threshold, noise_std, sin_amp, sin_amp_div);
//            the remaining S*H elements are the random noise values
//            that the CPU side seeded via std::vector<float> and
//            blitted through a GGML_TYPE_I32 tensor with
//            ggml_backend_tensor_set. Same bit layout on x86/RDNA so
//            we just reinterpret_cast to const float*.
static __global__ void k_uv_noise(
        float * __restrict__ dst,
        const float * __restrict__ gate,
        const float * __restrict__ params,
        int sequence_length,
        int harmonic_num) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= sequence_length) {
        return;
    }

    const float voice_threshold = params[0];
    const float noise_std       = params[1];
    const float sin_amp         = params[2];
    const float sin_amp_div     = params[3];
    const float * rand_init     = params + 4;

    const bool voiced  = (gate[r] > voice_threshold);
    const float uv     = voiced ? sin_amp   : 0.0f;
    const float scale  = voiced ? noise_std : sin_amp_div;

    const int sh   = sequence_length * harmonic_num;
    float * uv_dst    = dst;
    float * noise_dst = dst + sh;

    for (int h = 0; h < harmonic_num; ++h) {
        const int idx = h * sequence_length + r;
        uv_dst[idx]    = uv;
        noise_dst[idx] = scale * rand_init[idx];
    }
}

void ggml_cuda_op_uv_noise(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * gate   = dst->src[1];
    const ggml_tensor * params = dst->src[2];

    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(gate->type == GGML_TYPE_F32);
    // params' declared type is I32 upstream, but its contents are bit-cast
    // floats — see the comment in ggml_uv_noise() / kokoro_model.cpp.
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(gate));
    GGML_ASSERT(ggml_is_contiguous(params));

    const int sequence_length = (int) gate->ne[0];
    const int harmonic_num    = (int) dst->ne[1];
    GGML_ASSERT(dst->ne[0] == sequence_length);
    GGML_ASSERT(dst->ne[2] == 2);
    // params layout: [4 thresholds][S*H rand cells] -> 4 + S*H int32 slots.
    GGML_ASSERT(ggml_nelements(params) == (int64_t) 4 + (int64_t) sequence_length * harmonic_num);

    const float * gate_d   = (const float *) gate->data;
    const float * params_d = (const float *) params->data;
    float * dst_d          = (float *) dst->data;

    const dim3 block(128, 1, 1);
    const dim3 grid((sequence_length + block.x - 1) / block.x, 1, 1);
    k_uv_noise<<<grid, block, 0, ctx.stream()>>>(
        dst_d, gate_d, params_d, sequence_length, harmonic_num);
}

// ---------------------------------------------------------------------
// STFT / AA_STFT
// ---------------------------------------------------------------------
//
// CPU reference: ggml_compute_forward_stft_f32 in ggml-cpu.c (around
// line 2014, koboldcpp tree). The reference extracts a windowed +
// reflectively-padded segment of the time-domain signal for each frame,
// then runs a radix-2 real FFT via radix2_fft(mdst, phdst, ...). For
// AA_STFT it follows up with |z|/atan2 per bin.
//
// Shapes (from ggml_stft in ggml.c):
//   src0 (signal) : F32, shape [signal_len, batch, ...]
//   src1 (window) : F32, shape [n_fft] (window length == n_fft)
//   dst           : F32, shape [n_fft, n_frames, batch, 2]
//                   -- last dim stores (real/imag) or (abs/angle).
//   op_params     : [n_fft, hop] as int32_t.
//
// Kokoro uses n_fft = 1024. Per-frame cost of naive DFT is O(n_fft^2)
// which is ~1M multiplies per (frame, bin) thread pair, but n_frames is
// typically small (~O(tokens)) so the absolute wall-clock is well under
// the STFT/ISTFT budget for a 1–2 s utterance. Correctness-first, no
// rocFFT dependency.
//
// Launch pattern: one thread per (bin, frame, batch). Bins are the fast
// dim so memory writes coalesce across threadIdx.x.
static __global__ void k_stft(
        const float * __restrict__ src,     // [ne00, ne01, ...]
        const float * __restrict__ window,  // [n_fft]
        float       * __restrict__ dst,     // [n_fft, n_frames, batch, 2]
        const int  n_fft,
        const int  hop,
        const int  half,
        const int  ne00,             // input signal length
        const int  n_frames,
        const int  batch,
        const int  src_stride_batch, // in elements (nb01/sizeof(float))
        const int  dst_stride_frame, // in elements (nb1/sizeof(float))
        const int  dst_stride_batch, // in elements (nb2/sizeof(float))
        const int  dst_imag_stride,  // in elements (nb3/sizeof(float))
        const int  compute_abs_angle) {
    const int bin   = blockIdx.x * blockDim.x + threadIdx.x;
    const int frame = blockIdx.y;
    const int b     = blockIdx.z;
    if (bin >= n_fft || frame >= n_frames || b >= batch) {
        return;
    }

    const int ch = frame * hop;
    const float * tgt = src + b * src_stride_batch;

    // DFT twiddle: simple_dft uses k = -2*pi/n_fft * i * j.
    const float base_k = -2.0f * (float) M_PI / (float) n_fft;

    float sum_r = 0.0f;
    float sum_i = 0.0f;
    for (int j = 0; j < n_fft; j++) {
        // Reflective padding: mirror out-of-bounds indices back into the
        // signal so windows near the edges stay well-defined.
        int ai = ch - half + j;
        int idx;
        if (ai < 0) {
            idx = -ai;
        } else if (ai >= ne00) {
            idx = 2 * ne00 - ai - 1;
        } else {
            idx = ai;
        }
        const float v = tgt[idx] * window[j];

        const float k = base_k * (float) bin * (float) j;
        const float c = cosf(k);
        const float s = sinf(k);
        sum_r += v * c;
        sum_i += v * s;
    }

    float * out = dst + b * dst_stride_batch + frame * dst_stride_frame;
    if (compute_abs_angle) {
        const float abs_v = sqrtf(sum_r * sum_r + sum_i * sum_i);
        const float agl_v = atan2f(sum_i, sum_r);
        out[bin]                   = abs_v;
        out[bin + dst_imag_stride] = agl_v;
    } else {
        out[bin]                   = sum_r;
        out[bin + dst_imag_stride] = sum_i;
    }
}

static void ggml_cuda_op_stft_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool compute_abs_angle) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[0]  == sizeof(float));

    const int n_fft = ((const int32_t *) dst->op_params)[0];
    const int hop   = ((const int32_t *) dst->op_params)[1];
    const int half  = n_fft / 2;

    const int ne00     = (int) src0->ne[0];
    const int n_frames = (int) dst->ne[1];
    const int batch    = (int) dst->ne[2];

    // The CPU impl only supports window length == n_fft.
    GGML_ASSERT((int) src1->ne[0] == n_fft);
    GGML_ASSERT((int) dst->ne[0]  == n_fft);
    GGML_ASSERT((int) dst->ne[3]  == 2);

    const size_t es = sizeof(float);
    const int src_stride_batch = (int) (src0->nb[1] / es);
    const int dst_stride_frame = (int) (dst->nb[1]  / es);
    const int dst_stride_batch = (int) (dst->nb[2]  / es);
    const int dst_imag_stride  = (int) (dst->nb[3]  / es);

    const int block_x = 128;
    const dim3 block(block_x, 1, 1);
    const dim3 grid((n_fft + block_x - 1) / block_x,
                    (unsigned int) n_frames,
                    (unsigned int) MAX(batch, 1));

    k_stft<<<grid, block, 0, ctx.stream()>>>(
        (const float *) src0->data,
        (const float *) src1->data,
        (float *) dst->data,
        n_fft, hop, half,
        ne00, n_frames, MAX(batch, 1),
        src_stride_batch, dst_stride_frame, dst_stride_batch, dst_imag_stride,
        compute_abs_angle ? 1 : 0);
}

void ggml_cuda_op_stft(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_stft_impl(ctx, dst, /*compute_abs_angle=*/false);
}

void ggml_cuda_op_aa_stft(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_stft_impl(ctx, dst, /*compute_abs_angle=*/true);
}

// ---------------------------------------------------------------------
// ISTFT / AA_ISTFT
// ---------------------------------------------------------------------
//
// CPU reference: ggml_compute_forward_istft_f32 in ggml-cpu.c (~line
// 2142). For each frame it computes the real part of the inverse DFT
// of the per-frame spectrum, then overlap-adds into the time-domain
// output with the window re-applied. The CPU implements the iDFT via a
// forward FFT + index permutation (N-i mod N) trick; here we just do
// the iDFT directly — the arithmetic is identical, the loop layout
// differs.
//
// Input src0 shape  : [bins, n_frames, batch, 2]  (complex pair via nb3)
//                     bins == n_fft for the full-spectrum case,
//                     bins == n_fft/2 + 1 for the onesided case.
// Input src1        : window, [n_fft].
// Output dst shape  : [(n_frames-1)*hop, batch, 1, 1].
//
// AA_ISTFT: src0 contains (abs, angle) per bin; reconstruct complex as
// z = abs * (cos(angle) + j * mult * sin(angle)) where mult = -1 for
// mirrored bins (>half) under the onesided convention, else +1.
//
// NOTE: the CPU path has what looks like an unreachable corner in the
// non-abs-angle onesided branch where `ph = m` instead of `ph = phdst`.
// We preserve that behaviour literally so a graph that mixes CPU and
// CUDA backends produces bit-identical output. Kokoro only uses the
// abs-angle path, so this corner is cold in practice.
//
// Launch pattern: one thread per (n, frame, batch) where n is the
// sample index within the frame [0, n_fft). Each thread computes the
// real part of iDFT[n], applies the window, and atomically adds into
// the output at position (frame*hop + n - half).
static __global__ void k_istft(
        const float * __restrict__ src,    // [ne00, n_frames, batch, 2]
        const float * __restrict__ window, // [n_fft]
        float       * __restrict__ dst,    // [dst_length, batch]
        const int  n_fft,
        const int  hop,
        const int  half,
        const int  ne00,
        const int  n_frames,
        const int  batch,
        const int  dst_length,
        const int  src_stride_frame, // elements
        const int  src_stride_batch, // elements
        const int  src_imag_stride,  // elements (nb03 / sizeof(float))
        const int  dst_stride_batch, // elements
        const int  from_abs_angle) {
    const int n     = blockIdx.x * blockDim.x + threadIdx.x;
    const int frame = blockIdx.y;
    const int b     = blockIdx.z;
    if (n >= n_fft || frame >= n_frames || b >= batch) {
        return;
    }

    const bool onesided = (ne00 == half + 1);

    const float * src_real = src + b * src_stride_batch + frame * src_stride_frame;
    const float * src_imag = src_real + src_imag_stride;

    // Inverse DFT: x[n].real = (1/N) * sum_k (X_r[k]*cos - X_i[k]*sin).
    const float base_k = 2.0f * (float) M_PI / (float) n_fft;

    float sum = 0.0f;
    for (int k = 0; k < n_fft; k++) {
        int idx        = k;
        float multiplier = 1.0f;
        if (onesided && k >= half + 1) {
            idx        = n_fft - k;
            multiplier = -1.0f;
        }

        float m;
        float ph;
        if (from_abs_angle) {
            const float abs_v = src_real[idx];
            const float agl   = src_imag[idx];
            const float cs    = cosf(agl);
            const float sn    = sinf(agl);
            m  = abs_v * cs;
            ph = abs_v * multiplier * sn;
        } else if (onesided) {
            // Preserve CPU-impl quirk: non-AA onesided leaves ph = m
            // (the real part at idx) rather than the mirrored imag.
            // Cold path for Kokoro but we mirror it bit-for-bit.
            m  = src_real[idx];
            ph = m;
        } else {
            m  = src_real[idx];
            ph = src_imag[idx];
        }

        const float angle = base_k * (float) k * (float) n;
        const float c     = cosf(angle);
        const float s     = sinf(angle);
        sum += m * c - ph * s;
    }
    sum /= (float) n_fft;

    const int location = frame * hop + n - half;
    if (location >= 0 && location < dst_length) {
        const float w = window[n];
        atomicAdd(dst + b * dst_stride_batch + location, sum * w);
    }
}

static void ggml_cuda_op_istft_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool from_abs_angle) {
    const ggml_tensor * src0   = dst->src[0];
    const ggml_tensor * window = dst->src[1];

    GGML_ASSERT(src0->type   == GGML_TYPE_F32);
    GGML_ASSERT(window->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type    == GGML_TYPE_F32);
    GGML_ASSERT(src0->nb[0]   == sizeof(float));
    GGML_ASSERT(window->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[0]    == sizeof(float));

    const int n_fft = ((const int32_t *) dst->op_params)[0];
    const int hop   = ((const int32_t *) dst->op_params)[1];
    const int half  = n_fft / 2;

    const int ne00       = (int) src0->ne[0];
    const int n_frames   = (int) src0->ne[1];
    const int batch      = (int) src0->ne[2];
    const int dst_length = (int) dst->ne[0];

    GGML_ASSERT((int) window->ne[0] == n_fft);
    GGML_ASSERT(ne00 == n_fft || ne00 == half + 1);

    const size_t es = sizeof(float);
    const int src_stride_frame = (int) (src0->nb[1] / es);
    const int src_stride_batch = (int) (src0->nb[2] / es);
    const int src_imag_stride  = (int) (src0->nb[3] / es);
    const int dst_stride_batch = (int) (dst->nb[1]  / es);

    // dst is accumulated via atomicAdd; callers expect it zero'd first.
    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), ctx.stream()));

    const int block_x = 128;
    const dim3 block(block_x, 1, 1);
    const dim3 grid((n_fft + block_x - 1) / block_x,
                    (unsigned int) n_frames,
                    (unsigned int) MAX(batch, 1));

    k_istft<<<grid, block, 0, ctx.stream()>>>(
        (const float *) src0->data,
        (const float *) window->data,
        (float *) dst->data,
        n_fft, hop, half,
        ne00, n_frames, MAX(batch, 1), dst_length,
        src_stride_frame, src_stride_batch, src_imag_stride, dst_stride_batch,
        from_abs_angle ? 1 : 0);
}

void ggml_cuda_op_istft(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_istft_impl(ctx, dst, /*from_abs_angle=*/false);
}

void ggml_cuda_op_aa_istft(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_istft_impl(ctx, dst, /*from_abs_angle=*/true);
}

// ---------------------------------------------------------------------
// CONV_TRANSPOSE_1D_TTS
// ---------------------------------------------------------------------
//
// CPU reference: ggml_compute_forward_conv_transpose_1d_f{16,f32}_tts
// in ggml-cpu.c (around lines 2253 / 2346). This is a 1-D transposed
// convolution with groups support and explicit padding, stride,
// dilation=1. The F16 variant is the one Kokoro's istftnet vocoder
// hits; the F32 variant is kept for completeness (we template over the
// weight type).
//
// Tensor shapes (from ggml_conv_transpose_1d_tts in ggml.c):
//   src0 (kernel) : F16 or F32, shape [K, Cout/g0, Cin]   (ne00=K, ne01=Cout/g0, ne02=Cin)
//   src1 (signal) : F32,         shape [T, Cin]            (ne10=T,  ne11=Cin)
//   dst           : F32,         shape [out_len, Cout, 1, 1]
//   op_params     : [s0 (stride), p0 (pad), d0 (dilation, must be 1), g0 (groups)]
//
// The CPU F32 path writes to dst[i10*s0 + i00 - p0, cout]; the F16 path
// writes to dst[i10*s0 + i00, cout]. That difference looks like a CPU
// bug (p0 should be subtracted in both) but we keep the two write-index
// conventions in lockstep with the CPU so we don't silently diverge on
// graphs that run with p0 != 0 on CPU. In practice Kokoro uses p0 = 0
// for its transposed-conv blocks, which makes the two formulas
// identical anyway.
//
// Parallelisation: one thread per (output position, output channel)
// pair. Each thread iterates the K kernel offsets, derives the unique
// input time step i10 that maps to this output position (if any), and
// sums `gne02 = Cin / g0` input-channel contributions. Kokoro shapes
// here are small enough (Cout ~128, Cin ~128, K ~16, out_len ~25k) that
// this naive scheme is fine — a few hundred K kernel-weight / input-
// sample multiplies per thread.
template <typename W>
static __device__ __forceinline__ float ttscpp_to_f32(W v);

template <>
__device__ __forceinline__ float ttscpp_to_f32<float>(float v) { return v; }

template <>
__device__ __forceinline__ float ttscpp_to_f32<half>(half v) { return __half2float(v); }

template <typename W, bool SUBTRACT_P0>
static __global__ void k_conv_transpose_1d_tts(
        const W     * __restrict__ kernel,
        const float * __restrict__ input,
        float       * __restrict__ dst,
        const int K, const int Cout, const int Cin, const int T, const int ne0,
        const int s0, const int p0, const int gne02,
        const int kb00, const int kb01, const int kb02,   // kernel strides (elements)
        const int ib10, const int ib11,                    // input  strides (elements)
        const int ob1) {                                   // dst    row stride (elements)
    const int out_pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int cout    = blockIdx.y;
    if (out_pos >= ne0 || cout >= Cout) {
        return;
    }

    // Decompose the output channel into (group, in-group) exactly the
    // way the CPU impl does: `i1 / gne02` selects the first kernel input
    // channel that contributes, and kernel_cout is the kernel's out-
    // channel index. For the two Kokoro-relevant regimes:
    //   * standard (g0=1, gne02=Cin): kernel_cout = cout, cin_base = 0,
    //     so the inner loop walks all Cin kernel weights.
    //   * depthwise (gne02=1, ne01=1, Cout=Cin*g0): kernel_cout is
    //     pinned at 0 (there's only one kernel out-slot per group) and
    //     cin_base = cout, so each output channel pulls a single
    //     kernel tap.
    // Packing both into a branch-free form would need per-launch meta,
    // so we just switch on gne02 == 1 at runtime; the branch is warp-
    // uniform so there's no divergence cost.
    const int cin_base    = cout / gne02;
    const int kernel_cout = (gne02 == 1) ? 0 : cout;
    // Derive i10 from (out_pos, i00) so we only touch O(K * gne02) work
    // per thread instead of O(T * K).
    //   F32 CPU: out_pos == i10*s0 + i00 - p0  =>  i00 = out_pos + p0 - i10*s0
    //   F16 CPU: out_pos == i10*s0 + i00       =>  i00 = out_pos       - i10*s0
    // Equivalently, i10*s0 = out_pos + (SUBTRACT_P0 ? p0 : 0) - i00.
    const int shift = SUBTRACT_P0 ? p0 : 0;

    float acc = 0.0f;
    for (int i00 = 0; i00 < K; i00++) {
        const int numer = out_pos + shift - i00;
        if (numer < 0) {
            continue;
        }
        if (numer % s0 != 0) {
            continue;
        }
        const int i10 = numer / s0;
        if (i10 >= T) {
            continue;
        }
        // Match the CPU validity guard. Given we derived i10 from
        // out_pos these are never false in practice (see the note in
        // ttscpp_ops.cu), but we keep the check so any future shape
        // changes on the CPU side stay in sync.
        const bool ok = (i10 * s0 < p0 && i00 >= p0) ||
                        (i10 * s0 >= p0 && i10 * s0 + i00 - p0 < ne0);
        if (!ok) {
            continue;
        }

        for (int c = 0; c < gne02; c++) {
            const int cin = cin_base + c;
            if (cin >= Cin) {
                break;
            }
            const float w_val = ttscpp_to_f32<W>(kernel[cin * kb02 + kernel_cout * kb01 + i00 * kb00]);
            const float s_val = input [cin * ib11 + i10         * ib10];
            acc += w_val * s_val;
        }
    }
    dst[cout * ob1 + out_pos] = acc;
}

template <typename W, bool SUBTRACT_P0>
static void launch_conv_transpose_1d_tts(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        int s0, int p0, int g0) {
    const int K    = (int) src0->ne[0];
    const int Cout = (int) dst->ne[1];
    const int Cin  = (int) src0->ne[2];
    const int T    = (int) src1->ne[0];
    const int ne0  = (int) dst->ne[0];
    const int gne02 = Cin / g0;

    const size_t ew = sizeof(W);
    const size_t sw = sizeof(float);

    const int kb00 = (int) (src0->nb[0] / ew);
    const int kb01 = (int) (src0->nb[1] / ew);
    const int kb02 = (int) (src0->nb[2] / ew);
    const int ib10 = (int) (src1->nb[0] / sw);
    const int ib11 = (int) (src1->nb[1] / sw);
    const int ob1  = (int) (dst->nb[1]  / sw);

    const int block_x = 128;
    const dim3 block(block_x, 1, 1);
    const dim3 grid((ne0 + block_x - 1) / block_x, (unsigned int) Cout, 1);

    k_conv_transpose_1d_tts<W, SUBTRACT_P0><<<grid, block, 0, ctx.stream()>>>(
        (const W *) src0->data,
        (const float *) src1->data,
        (float *) dst->data,
        K, Cout, Cin, T, ne0,
        s0, p0, gne02,
        kb00, kb01, kb02,
        ib10, ib11,
        ob1);
}

void ggml_cuda_op_conv_transpose_1d_tts(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int32_t * opts = (const int32_t *) dst->op_params;
    const int s0 = opts[0];
    const int p0 = opts[1];
    const int d0 = opts[2];
    const int g0 = opts[3];
    GGML_ASSERT(d0 == 1);
    GGML_UNUSED(d0);

    if (src0->type == GGML_TYPE_F16) {
        launch_conv_transpose_1d_tts<half, /*SUBTRACT_P0=*/false>(ctx, src0, src1, dst, s0, p0, g0);
    } else if (src0->type == GGML_TYPE_F32) {
        launch_conv_transpose_1d_tts<float, /*SUBTRACT_P0=*/true>(ctx, src0, src1, dst, s0, p0, g0);
    } else {
        GGML_ABORT("conv_transpose_1d_tts: unsupported kernel dtype");
    }
}

// ---------------------------------------------------------------------
// UPSCALE_LINEAR
// ---------------------------------------------------------------------
//
// CPU reference: ggml_compute_forward_upscale_linear_f32 in ggml-cpu.c
// (~line 2461). 1-D linear interpolation upsampling by an integer
// factor, with PyTorch-style half-scale-factor padding on each end.
//
// Inputs:
//   src0 : F32, arbitrary 4-D tensor [ne00, ne01, ne02, ne03].
//   dst  : F32, shape [ne00 * scale, ne01, ne02, ne03].
//
// The scale factor is implicit: sf0 = ne0 / src0->ne[0] (ne0 is dst's
// first dim). We treat this as one thread per output element — the
// arithmetic is essentially a few multiplies per thread, so even
// launching ne0 * ne1 * ne2 * ne3 threads is cheap.
static __global__ void k_upscale_linear(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        const int ne0_src, const int ne0_dst,
        const int ne1, const int ne2, const int ne3,
        const int s00, const int s01, const int s02, const int s03,   // src strides (elements)
        const int d0,  const int d1,  const int d2,  const int d3,    // dst strides (elements)
        const float sf0, const float hsf0, const int sf, const int hsf) {
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int i1 = blockIdx.y;
    const int ii = blockIdx.z;
    const int i2 = ii % ne2;
    const int i3 = ii / ne2;
    if (i0 >= ne0_dst || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const float * row_src = src + i1 * s01 + i2 * s02 + i3 * s03;
    float       * row_dst = dst + i1 * d1  + i2 * d2  + i3 * d3;

    float v;
    if (i0 < hsf) {
        v = row_src[0];
    } else if (i0 >= ne0_dst - hsf) {
        v = row_src[(ne0_src - 1) * s00];
    } else {
        // Float division truncated to int matches CPU behaviour
        // (const int64_t i00 = (i0 - hsf0) / sf0;).
        const int i00 = (int) (((float) i0 - hsf0) / sf0);
        const float base = row_src[ i00      * s00];
        const float top  = row_src[(i00 + 1) * s00];
        const float diff_adj = (top - base) / sf0;
        const int   mod_v    = (i0 - hsf) % sf;
        const float adj      = (float) mod_v * diff_adj + diff_adj * 0.5f;
        v = base + adj;
    }
    row_dst[i0 * d0] = v;
}

void ggml_cuda_op_upscale_linear(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int ne0_src = (int) src0->ne[0];
    const int ne0_dst = (int) dst->ne[0];
    const int ne1     = (int) dst->ne[1];
    const int ne2     = (int) dst->ne[2];
    const int ne3     = (int) dst->ne[3];
    GGML_ASSERT(dst->ne[1] == src0->ne[1]);
    GGML_ASSERT(dst->ne[2] == src0->ne[2]);
    GGML_ASSERT(dst->ne[3] == src0->ne[3]);
    GGML_ASSERT(ne0_src > 0);

    const float sf0  = (float) ne0_dst / (float) ne0_src;
    const float hsf0 = sf0 / 2.0f;
    const int   sf   = (int) sf0;
    const int   hsf  = (int) hsf0;
    // The CPU impl uses `(i0 - hsf) % sf` which is UB for sf == 0.
    // Scale-down isn't a supported configuration (see comment in
    // ggml-cpu.c). Assert rather than silently produce garbage.
    GGML_ASSERT(sf >= 1);

    const size_t es = sizeof(float);
    const int s00 = (int) (src0->nb[0] / es);
    const int s01 = (int) (src0->nb[1] / es);
    const int s02 = (int) (src0->nb[2] / es);
    const int s03 = (int) (src0->nb[3] / es);
    const int d0  = (int) (dst->nb[0]  / es);
    const int d1  = (int) (dst->nb[1]  / es);
    const int d2  = (int) (dst->nb[2]  / es);
    const int d3  = (int) (dst->nb[3]  / es);

    const int block_x = 128;
    const dim3 block(block_x, 1, 1);
    const dim3 grid((ne0_dst + block_x - 1) / block_x,
                    (unsigned int) ne1,
                    (unsigned int) (ne2 * ne3));

    k_upscale_linear<<<grid, block, 0, ctx.stream()>>>(
        (const float *) src0->data,
        (float *) dst->data,
        ne0_src, ne0_dst,
        ne1, ne2, ne3,
        s00, s01, s02, s03,
        d0,  d1,  d2,  d3,
        sf0, hsf0, sf, hsf);
}
