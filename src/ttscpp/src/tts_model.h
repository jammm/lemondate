#ifndef tts_model_h
#define tts_model_h

#include <cstring>
#include <functional>
#include "ttsutil.h"
#include "ttscommon.h"

void append_to_response(struct tts_response * response, struct tts_response * to_append);

using tensor_meta_callback = std::function<void(ggml_tensor*)>*;

struct runner_context {
    runner_context(int n_threads): n_threads(n_threads) {};
    virtual ~runner_context() {
        ggml_backend_sched_free(sched);
        // `backend_cpu` + `threadpool` are dead state in lemondate
        // (GGML_CPU=OFF). `backend_cpu` is never initialized because
        // ggml_backend_cpu_init is not linked; `threadpool` likewise
        // stays null. The fields are preserved to avoid a ripple of
        // edits across every reader, but the corresponding
        // ggml_backend_free / ggml_threadpool_free calls would pull
        // in CPU-backend symbols and are omitted.
        //
        // Skip freeing `backend` when we don't own it — kokoro shares
        // one CUDA/HIP backend across model + both kctx instances to
        // keep the HIP stream count down to one (multiple
        // ggml_backend_cuda_init calls observed to crash on gfx1201).
        // The owning party (kokoro_model::free) handles the free.
        if (owned_backend) {
            ggml_backend_free(backend);
        }
        ggml_backend_buffer_free(buf_output);
    }
    // TODO: extend the backend and buffer support out to all devices
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t backend_buffer = nullptr;
    // True when this context allocated `backend` and is responsible
    // for freeing it. False when `backend` is borrowed from another
    // owner (e.g. kokoro_model on the GPU path).
    bool owned_backend = true;

    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_type_t backend_cpu_buffer = nullptr;

    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_buffer_t buf_output = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_threadpool_t threadpool = nullptr;
    float * logits = nullptr;
    int n_threads;

    void get_ggml_node_data(struct ggml_tensor * output_tensor, float * output, size_t output_size, ggml_backend_buffer_t buffer = nullptr);
    void set_threads();
    void build_schedule(size_t max_nodes);
    bool prep_schedule(ggml_cgraph * gf);
    void prep_output_buffer(size_t new_size);
};

struct tts_model {
    struct model_tensor_meta tensor_meta;

    // this is the current byte offset into the model's buffer.
    size_t offset = 0;

    bool use_cross_attn = true;

    ggml_backend_buffer_type_t buffer = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    // it is quite common for implementations of tts_model to need to update attributes or perform distinct operations
    // when computing the tensor meta of the loaded model. This callback allows this as it will receive each processed tensor.
    tensor_meta_callback compute_tensor_meta_cb = nullptr;

    struct ggml_context * ctx;

    void prep_buffers_and_context(bool cpu_only, float size_offset, uint32_t dedicated_add_on_size);
    void setup_from_file(gguf_context * meta_ctx, ggml_context * load_context, bool cpu_only, std::string model_prefix, float size_offset = 1.4, uint32_t dedicated_add_on_size = 0);
    void set_tensor(struct ggml_tensor * tensor, struct ggml_tensor * target);
    size_t max_nodes();
    void assign_weight(std::string name, ggml_tensor * tensor);
    void free();
};

#endif
