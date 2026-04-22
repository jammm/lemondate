#include "tts_model.h"
#include "ggml-backend.h"

void append_to_response(struct tts_response * response, struct tts_response * to_append) {
    float * new_data = (float *) malloc((response->n_outputs + to_append->n_outputs) * sizeof(float));
    if (response->n_outputs > 0) {
        std::memcpy(new_data, response->data, response->n_outputs*sizeof(float));
    }
    if (to_append->n_outputs > 0) {
        float * next_loc = new_data + response->n_outputs;
        std::memcpy(next_loc, to_append->data, to_append->n_outputs*sizeof(float));
    }
    response->data = new_data;
    response->n_outputs += to_append->n_outputs;
}

/*
 * Pulls output_size to prepped buffer 'output' from 'output_node' tensor. If no buffer is passed will default to the existing output buffer present
 * on runner_context.
 */
void runner_context::get_ggml_node_data(struct ggml_tensor * output_node, float * output, size_t output_size, ggml_backend_buffer_t buffer) {
    if (buffer == nullptr) {
        buffer = buf_output;
    }
    if (ggml_backend_buffer_get_size(buffer) < output_size) {
        TTS_ABORT("Output buffer overflow of %d / %d for output node '%s'\n", output_size, ggml_backend_buffer_get_size(buffer), ggml_get_name(output_node));
    } else if (ggml_nbytes(output_node) < output_size) {
        TTS_ABORT("Output node, '%s', with %d bytes is too small for #ggml_backend_tensor_get_async with size of %d.\n", ggml_get_name(output_node), ggml_nbytes(output_node), output_size);
    }
    ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched, output_node);
    ggml_backend_tensor_get_async(backend_res, output_node, output, 0, output_size);
}

void runner_context::set_threads() {
    // No-op in lemondate: thread-pool + thread-count management is
    // part of the ggml CPU compute backend, which this build does not
    // link against (GGML_CPU=OFF). All compute runs on the HIP/CUDA
    // backend; anything that needs threads uses the GPU device queue.
}

void runner_context::build_schedule(size_t max_nodes) {
    // `backend_cpu_buffer` is a host-memory buffer *type* descriptor
    // (ggml-base, not ggml-cpu) used by prep_output_buffer() and by
    // the kokoro runner to allocate the final audio-sample readback
    // buffer. It does not require the CPU compute backend. If the
    // linker rejects ggml_backend_cpu_buffer_type with GGML_CPU=OFF
    // the fallback is ggml_backend_cuda_host_buffer_type(0), which
    // ships with ggml-cuda / ggml-hip.
    backend_cpu_buffer = ggml_backend_cpu_buffer_type();
    // GPU-only scheduler. Lemondate builds ggml with GGML_CPU=OFF so
    // there is no CPU compute backend to register as a fallback. The
    // `backend` member must be set to a HIP/CUDA backend by the owner
    // (kokoro_from_file wires model->backend and hands it to every
    // runner_context) before build_schedule runs.
    TTS_ASSERT(backend != nullptr);
    std::vector<ggml_backend_buffer_type_t> bufs = {backend_buffer};
    std::vector<ggml_backend_t> backs = {backend};
    // op_offload=true: push large matmuls onto the primary backend
    // even when their inputs are placed elsewhere. With a single GPU
    // backend this is effectively a no-op for placement but keeps the
    // scheduler behavior consistent with the previous CPU+GPU setup.
    sched = ggml_backend_sched_new(backs.data(), bufs.data(), 1, max_nodes, false, true);
}

bool runner_context::prep_schedule(struct ggml_cgraph * gf) {
    return ggml_backend_sched_reserve(sched, gf);
}

void runner_context::prep_output_buffer(size_t new_size) {
    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output) : 0;
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
            ggml_backend_buffer_free(buf_output);
            buf_output = nullptr;
            logits = nullptr;
        }
        buf_output = ggml_backend_buft_alloc_buffer(backend_cpu_buffer, new_size);
    }
    logits = (float *) ggml_backend_buffer_get_base(buf_output);
}

void tts_runner::init_build(std::vector<uint8_t>* buf_compute_meta) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta->size(),
        /*.mem_buffer =*/ buf_compute_meta->data(),
        /*.no_alloc   =*/ true,
    };

    ctx = ggml_init(params);
}

void tts_runner::free_build() {
    if (ctx) {
        ggml_free(ctx);
        ctx = nullptr;
    }
}

void tts_model::prep_buffers_and_context(bool cpu_only, float size_offset, uint32_t dedicated_add_on_size) {
    // Lemondate ships a GPU-only ttscpp. `cpu_only=true` was the path
    // that called ggml_backend_cpu_init() / ggml_backend_cpu_buffer_
    // type() — both live in the ggml CPU compute backend which this
    // build does not include (GGML_CPU=OFF). There is therefore no
    // sensible CPU fallback and we abort rather than silently produce
    // a model with a null backend.
    if (cpu_only) {
        TTS_ABORT("ttscpp was built GPU-only (GGML_CPU=OFF); cpu_only=true is not supported. "
                  "Set cpu_only=false and ensure kokoro_from_file wired model->backend to a HIP/CUDA backend.");
    }
    if (!backend || !buffer) {
        TTS_ABORT("tts_model::prep_buffers_and_context: model->backend / model->buffer must be set "
                  "by the caller (e.g. kokoro_from_file) before setup_from_file in a GPU-only build.");
    }
    size_t ctx_size = ggml_tensor_overhead() * (tensor_meta.n_tensors * size_offset);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    if(dedicated_add_on_size>13000)
    {
        printf("Clamp TTS addon memory %zu to 13000\n",(size_t)dedicated_add_on_size);
        dedicated_add_on_size = 13000;
    }
    // Pad the buffer for per-tensor alignment in set_tensor(). The
    // CUDA / HIP backend asks for 128-byte alignment per tensor; CPU
    // backends are happy at 1 byte. We over-allocate by
    // (n_tensors * (alignment-1)) so set_tensor's round-up-to-align
    // can never run past the end of the buffer. Worst case is ~80 KB
    // for a 600-tensor model — negligible vs the 200 MB Q4 weights.
    const size_t buf_alignment = ggml_backend_buft_get_alignment(buffer);
    const size_t alignment_pad = tensor_meta.n_tensors * (buf_alignment > 0 ? buf_alignment - 1 : 0);
    printf("TTS Memory Requested: %zu, with buffer %zu + %zu (+ %zu alignment pad, align=%zu)\n",
        ctx_size, tensor_meta.n_bytes, (size_t)dedicated_add_on_size, alignment_pad, buf_alignment);
    ctx = ggml_init(params);
    buf = ggml_backend_buft_alloc_buffer(buffer,
        tensor_meta.n_bytes + dedicated_add_on_size + alignment_pad);
}

void tts_model::assign_weight(std::string name, ggml_tensor * tensor) {
	TTS_ABORT("%s received name, %s, tensor without being defined. %s must be defined for all implementations of tts_model. \n", __func__, name.c_str(), __func__);
}

void tts_model::set_tensor(struct ggml_tensor * tensor, struct ggml_tensor * target) {
    // Round offset up to the buffer's required alignment before placing
    // the tensor. The CUDA/HIP backend wants 128-byte alignment for
    // each tensor; passing it a misaligned device pointer produces
    // "ROCm error: unspecified launch failure" on the very first
    // kernel that touches the tensor (we hit this on
    // kokoro.decoder.generator.noise_blocks.0.resblock.0.alpha1 with a
    // pointer ending in 0x1E). The CPU backend reports alignment=1 so
    // this is a no-op for CPU loads.
    const size_t align = ggml_backend_buft_get_alignment(buffer);
    if (align > 1) {
        offset = (offset + align - 1) & ~(align - 1);
    }
    tensor->buffer = buf;
    tensor->data = (void *)((uint8_t *) ggml_backend_buffer_get_base(buf) + offset);
    size_t size = ggml_nbytes(target);
    ggml_backend_tensor_set(tensor, target->data, 0, size);
    ggml_set_name(tensor, target->name);
    offset += size;
}

void tts_model::setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only, std::string model_prefix, float size_offset, uint32_t dedicated_add_on_size) {
    tensor_meta = compute_tensor_meta(model_prefix, load_context, compute_tensor_meta_cb);
    prep_buffers_and_context(cpu_only, size_offset, dedicated_add_on_size);
}

size_t tts_model::max_nodes() {
    return std::max<size_t>(8192, tensor_meta.n_tensors*5);
}

void tts_model::free() {
    if (ctx) {
        ggml_free(ctx);
    }
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
    if (backend) {
        ggml_backend_free(backend);
    }
}
