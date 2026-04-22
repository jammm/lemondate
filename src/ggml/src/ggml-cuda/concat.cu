#include "concat.cuh"

// contiguous kernels
//
// All three dim0/dim1/dim2 kernels take a y_off / ne1 pair so the host
// can tile along the ne1 dimension for tensors whose ne1 exceeds HIP's
// 65535 grid.y cap. Kokoro's ISTFTNet decoder (n_fft=20 / hop=5)
// produces STFT outputs with ne1 in the 60-80k range for any TTS
// response past ~13.6s of audio, which blew up this kernel's launch
// (grid.y = ne1) until the tiling went in. ne1 is the *full* height
// so offset math using the ne2 stride stays correct after tiling;
// y_off is the base index for this launch's grid.y block.
static __global__ void concat_f32_dim0(const float * x, const float * y, float * dst,
                                       const int ne0, const int ne00,
                                       const int ne1, const int y_off) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }
    // "yy" not "y" to avoid shadowing the src1 pointer argument above.
    const int yy = y_off + (int) blockIdx.y;
    if (yy >= ne1) {
        return;
    }

    int offset_dst =
        nidx +
        yy * ne0 +
        blockIdx.z * ne0 * ne1;

    if (nidx < ne00) { // src0
        int offset_src =
            nidx +
            yy * ne00 +
            blockIdx.z * ne00 * ne1;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            (nidx - ne00) +
            yy * (ne0 - ne00) +
            blockIdx.z * (ne0 - ne00) * ne1;
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim1(const float * x, const float * y, float * dst,
                                       const int ne0, const int ne01,
                                       const int ne1, const int y_off) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }
    const int yy = y_off + (int) blockIdx.y;
    if (yy >= ne1) {
        return;
    }

    int offset_dst =
        nidx +
        yy * ne0 +
        blockIdx.z * ne0 * ne1;

    if (yy < ne01) { // src0
        int offset_src =
            nidx +
            yy * ne0 +
            blockIdx.z * ne0 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            (yy - ne01) * ne0 +
            blockIdx.z * ne0 * (ne1 - ne01);
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim2(const float * x, const float * y, float * dst,
                                       const int ne0, const int ne02,
                                       const int ne1, const int y_off) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }
    const int yy = y_off + (int) blockIdx.y;
    if (yy >= ne1) {
        return;
    }

    int offset_dst =
        nidx +
        yy * ne0 +
        blockIdx.z * ne0 * ne1;

    if (blockIdx.z < (unsigned)ne02) { // src0
        int offset_src =
            nidx +
            yy * ne0 +
            blockIdx.z * ne0 * ne1;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            yy * ne0 +
            (blockIdx.z - ne02) * ne0 * ne1;
        dst[offset_dst] = y[offset_src];
    }
}

static void concat_f32_cuda(const float * x, const float * y, float * dst,
                            int ne00, int ne01, int ne02,
                            int ne0, int ne1, int ne2, int dim, cudaStream_t stream) {
    const int num_blocks = (ne0 + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
    // Tile along ne1 to stay under HIP's 65535 grid.y cap.
    const int rows_per_launch = 32768;
    for (int y_off = 0; y_off < ne1; y_off += rows_per_launch) {
        const int this_rows = std::min(rows_per_launch, ne1 - y_off);
        dim3 gridDim(num_blocks, this_rows, ne2);
        if (dim == 0) {
            concat_f32_dim0<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                x, y, dst, ne0, ne00, ne1, y_off);
        } else if (dim == 1) {
            concat_f32_dim1<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                x, y, dst, ne0, ne01, ne1, y_off);
        } else {
            concat_f32_dim2<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                x, y, dst, ne0, ne02, ne1, y_off);
        }
    }
}

// non-contiguous kernel (slow)
template <int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_f32_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3){
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const float * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const float *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const float *) (src1 + i3 * nb13 + i2 * nb12 + i1 * nb11 + (i0 - ne00) * nb10);
            } else if constexpr (dim == 1) {
                x = (const float *) (src1 + i3 * nb13 + i2 * nb12 + (i1 - ne01) * nb11 + i0 * nb10);
            } else if constexpr (dim == 2) {
                x = (const float *) (src1 + i3 * nb13 + (i2 - ne02) * nb12 + i1 * nb11 + i0 * nb10);
            } else if constexpr (dim == 3) {
                x = (const float *) (src1 + (i3 - ne03) * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);
            }
        }

        float * y = (float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}


void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const float * src0_d = (const float *)src0->data;
        const float * src1_d = (const float *)src1->data;

        float * dst_d = (float *)dst->data;

        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_f32_cuda(
                        src0_d + i3 * (src0->nb[3] / 4),
                        src1_d + i3 * (src1->nb[3] / 4),
                        dst_d + i3 * ( dst->nb[3] / 4),
                        src0->ne[0], src0->ne[1], src0->ne[2],
                        dst->ne[0],  dst->ne[1],  dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            CUDA_CHECK(cudaMemcpyAsync(dst_d,           src0_d, size0, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dst_d + size0/4, src1_d, size1, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            concat_f32_non_cont<dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}
