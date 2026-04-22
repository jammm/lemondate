#include "ttscpp.h"
#include <mutex>
// Needed for ggml_backend_cuda_init / ggml_backend_cuda_buffer_type. The
// header is compiled whenever GGML_USE_CUDA or GGML_USE_HIP is defined,
// and koboldcpp's HIPBLAS build sets both.
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

// A list of all of the top level GGUF names under kokoro.duration_predictor that have quantization compatible tensors.
static constexpr std::array<const char *, 5> DURATION_PREDICTOR_QUANTIZATION_COMPATIBLE_PARTS = {
    "duration_proj",
    "encode",
    "shared_lstm",
    "duration_lstm",
    "layers"
};

struct tts_runner * kokoro_from_file(gguf_context * meta_ctx, ggml_context * weight_ctx, int n_threads, generation_configuration * config, tts_arch arch, bool cpu_only) {
    kokoro_model * model = new kokoro_model;
    single_pass_tokenizer * spt = single_pass_tokenizer_from_gguf(meta_ctx, "tokenizer.ggml.tokens");
    // GPU path. Pre-set the model's backend/buffer so
    // tts_model::prep_buffers_and_context takes the non-cpu branch and
    // loads weights straight into device memory. Then both kokoro
    // contexts (duration + audio) reuse model->backend (rather than
    // calling ggml_backend_cuda_init separately) so we have ONE HIP
    // context with ONE stream — multiple ggml_backend_cuda_init() calls
    // produce independent stream objects whose interleaving was
    // observed to crash mid-graph with "ROCm error: unspecified launch
    // failure" on gfx1201. The set_inputs() helpers in kokoro_model.cpp
    // were rewritten to use CPU staging buffers + ggml_backend_tensor_
    // set so they work regardless of buffer location, and
    // post_load_assign caches the iSTFT window on CPU (model->window_
    // cpu_cache) for the CPU-only compute_window_squared_sum helper.
#ifdef GGML_USE_CUDA
    if (!cpu_only) {
        ggml_backend_t gpu_backend = ggml_backend_cuda_init(0);
        if (gpu_backend != nullptr) {
            model->backend = gpu_backend;
            model->buffer  = ggml_backend_cuda_buffer_type(0);
            fprintf(stdout, "Kokoro: CUDA/HIP backend initialized (device=0)\n");
        } else {
            fprintf(stdout, "Kokoro: CUDA/HIP init failed - falling back to CPU\n");
            cpu_only = true;
        }
    }
#else
    cpu_only = true;
#endif
    model->setup_from_file(meta_ctx, weight_ctx, cpu_only);
    struct kokoro_duration_context * kdctx = build_new_duration_kokoro_context(model, n_threads, cpu_only);
    struct kokoro_duration_runner * duration_runner = new kokoro_duration_runner(model, kdctx, spt);
    struct kokoro_context * kctx = build_new_kokoro_context(model, n_threads, cpu_only);
    // if an espeak voice id wasn't specifically set infer it from the kokoro voice, if it was override it, otherwise fallback to American English.
    std::string espeak_voice_id = config->espeak_voice_id;
    if (espeak_voice_id.empty()) {
        espeak_voice_id = !config->voice.empty() && KOKORO_LANG_TO_ESPEAK_ID.find(config->voice.at(0)) != KOKORO_LANG_TO_ESPEAK_ID.end() ? KOKORO_LANG_TO_ESPEAK_ID[config->voice.at(0)] : "gmw/en-US";
    }
    struct phonemizer * phmzr = phonemizer_from_gguf(meta_ctx, espeak_voice_id);
    struct kokoro_runner * runner = new kokoro_runner(model, kctx, spt, duration_runner, phmzr);

    // TODO: change this weight assignment pattern to mirror llama.cpp
    for (ggml_tensor * cur = ggml_get_first_tensor(weight_ctx); cur; cur = ggml_get_next_tensor(weight_ctx, cur)) {
        runner->assign_weight(cur->name, cur);
    }

    runner->prepare_post_load();

    gguf_free(meta_ctx);
    ggml_free(weight_ctx);
    runner->arch = arch;

    return (tts_runner*)runner;
}

// Lemondate ships Kokoro only. Parler / Dia / Orpheus code paths were
// removed with their model sources; any other architecture in the GGUF
// is a hard error.
struct tts_runner * runner_from_file(const std::string & fname, int n_threads, generation_configuration * config, bool cpu_only) {
    ggml_context * weight_ctx = NULL;

    struct gguf_init_params params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &weight_ctx,
    };
    gguf_context * meta_ctx = gguf_init_from_file(fname.c_str(), params);
    if (!meta_ctx) {
        fprintf(stdout,"%s failed for file %s\n", __func__, fname.c_str());
        return nullptr;
    }
    int arch_key = gguf_find_key(meta_ctx, "general.architecture");
    if (arch_key == -1) {
        fprintf(stdout,"%s failed for file %s. No architecture is set.\n", __func__, fname.c_str());
        return nullptr;
    }
    std::string arch = std::string(gguf_get_val_str(meta_ctx, arch_key));
    if (TTSCPP_SUPPORTED_ARCHITECTURES.find(arch) == TTSCPP_SUPPORTED_ARCHITECTURES.end()) {
        fprintf(stdout,"%s failed for file %s. The architecture '%s' is not supported.", __func__, fname.c_str(), arch.c_str());
        return nullptr;
    }
    tts_arch arch_type = TTSCPP_SUPPORTED_ARCHITECTURES.at(arch);
    if (arch_type != KOKORO_ARCH) {
        TTS_ABORT("Only kokoro arch is supported in this build");
    }
    return kokoro_from_file(meta_ctx, weight_ctx, n_threads, config, arch_type, cpu_only);
}

//returns 0 on success
int generate(tts_runner * runner, std::string sentence, struct tts_response * response, generation_configuration * config) {
    switch(runner->arch) {
        case KOKORO_ARCH:
            return ((kokoro_runner*)runner)->generate(sentence, response, config->voice, config->espeak_voice_id);
        default:
            TTS_ABORT("%s failed. The architecture '%d' is not supported.", __func__, runner->arch);
    }
}

std::vector<std::string> list_voices(tts_runner * runner) {
    switch(runner->arch) {
        case KOKORO_ARCH:
            return ((kokoro_runner*)runner)->list_voices();
        default:
            TTS_ABORT("%s failed. The architecture '%d' does not support #list_voices supported.", __func__, runner->arch);
    }
}

bool kokoro_is_f16_compatible(std::string name) {
    return name.find("voice_tensors") == std::string::npos &&
           name.find("bias") == std::string::npos &&
           name.find("gamma") == std::string::npos &&
           name.find("beta") == std::string::npos &&
           name.find("alpha") == std::string::npos &&
           !has_suffix(name, "embd") &&
           !has_suffix(name, "norm");
}

bool kokoro_is_quantizable(std::string name, struct quantization_params * params) {
    if (kokoro_is_f16_compatible(name)) {
        if (has_prefix(name, "kokoro.albert") || has_prefix(name, "kokoro.text_encoder.lstm")) {
            return true;
        } else if (has_prefix(name, "kokoro.duration_predictor.")) {
            std::vector<std::string> parts = split(name, ".");
            for (std::string part : DURATION_PREDICTOR_QUANTIZATION_COMPATIBLE_PARTS) {
                if (part == parts[2]) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool is_quantizable(tts_arch arch, std::string name, struct quantization_params * params) {
    switch(arch) {
        case KOKORO_ARCH:
            return kokoro_is_quantizable(name, params);
        default:
            TTS_ABORT("%s failed. The architecture '%d' is not supported.", __func__, arch);
    }
}

size_t quantize_tensor(void * new_data, struct ggml_tensor * tensor, const float * imatrix, enum ggml_type qtype, uint32_t n_threads) {
    // much of this is form copied from llama.cpp
    int chunk_size_multiplier = 1;
    if (qtype == GGML_TYPE_Q4_0_4_4 || qtype == GGML_TYPE_Q4_0_4_8 || qtype == GGML_TYPE_Q4_0_8_8) {
        if ((qtype == GGML_TYPE_Q4_0_8_8) && (tensor->ne[1] % 8 != 0)) qtype = GGML_TYPE_Q4_0;
        else if (tensor->ne[1] % 4 != 0) qtype = GGML_TYPE_Q4_0;
        if (qtype == GGML_TYPE_Q4_0_8_8) chunk_size_multiplier = 8;
        else if (qtype == GGML_TYPE_Q4_0_4_4 || qtype == GGML_TYPE_Q4_0_4_8) chunk_size_multiplier = 4;
    }
    size_t out_size = 0;
    const int32_t d3_step = tensor->ne[0] * tensor->ne[1];
    const int32_t n_per_row = tensor->ne[0];
    const int32_t nrows = tensor->ne[1];
    static const int32_t min_chunk_size = 32 * 512;
    const int32_t chunk_size = (n_per_row >= min_chunk_size ? n_per_row : n_per_row * ((min_chunk_size + n_per_row - 1)/n_per_row)) * chunk_size_multiplier;
    uint32_t thread_count = std::max(1, std::min((int)n_threads, (int)(d3_step + chunk_size - 1) / chunk_size));
    std::mutex mutex;

    for (int32_t d3_index = 0; d3_index < tensor->ne[2]; d3_index++) {
        const float * f32_data_d3 = ((float *) tensor->data) + d3_index * d3_step;
        void * new_data_d3 = (char *)new_data + ggml_row_size(qtype, tensor->ne[0]) * d3_index * nrows;
        const float * imatrix_03 = imatrix ? imatrix + d3_index * tensor->ne[0] : nullptr;
        if (thread_count <= 1) {
            // not threaded
            out_size += ggml_quantize_chunk(qtype, f32_data_d3, new_data_d3, 0, nrows, n_per_row, imatrix);
        } else {
            std::vector <std::thread> threads;
            int64_t counter = 0;
            size_t new_size = 0;
            bool valid = true;
            for (uint32_t t = 0; t < thread_count; t++) {
                auto func = [&mutex, &counter, &new_size, &valid, qtype, f32_data_d3, new_data_d3, chunk_size, nrows, n_per_row, imatrix]() {
                    const int64_t nrows_per_chunk = chunk_size / n_per_row;
                    size_t local_size = 0;
                    while (true) {
                        std::unique_lock<std::mutex> lock(mutex);
                        int64_t first_row = counter;
                        counter += nrows_per_chunk;
                        if (first_row >= nrows) {
                            if (local_size > 0) {
                                new_size += local_size;
                            }
                            break;
                        }
                        lock.unlock();
                        const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);
                        size_t this_size = ggml_quantize_chunk(qtype, f32_data_d3, new_data_d3, first_row * n_per_row, this_nrow, n_per_row, imatrix);
                        local_size += this_size;

                        // validate the quantized data; I am not sure how this would occur, but there is always the safe fallback on doing this single threaded.
                        const size_t row_size  = ggml_row_size(qtype, n_per_row);
                        void * this_data = (char *) new_data_d3 + first_row * row_size;
                        if (!ggml_validate_row_data(qtype, this_data, this_size)) {
                            std::unique_lock<std::mutex> lock(mutex);
                            valid = false;
                            break;
                        }
                    }
                };
                threads.push_back(std::thread(func));
            }
            for (auto & t : threads) t.join();

            if (!valid) {
                TTS_ABORT("Validation of quantized data failed. Please try again and/or switch to single thread quantization.\n");
            }
            out_size += new_size;
        }
    }
    return out_size;
}

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

template <typename T>
struct do_no_init {
    T value;
    do_no_init() { /* do nothing */ }
};

void quantize_gguf(const std::string & ifile, const std::string & ofile, struct quantization_params * params) {
    ggml_context * weight_ctx = NULL;
    struct gguf_init_params gguf_params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &weight_ctx,
    };
    gguf_context * meta_ctx = gguf_init_from_file(ifile.c_str(), gguf_params);
    std::string arch = "parler-tts"; // only parler-tts gguf files should lack an explicit architecture.

    int arch_key = gguf_find_key(meta_ctx, "general.architecture");
    if (arch_key != -1) {
        arch = std::string(gguf_get_val_str(meta_ctx, arch_key));
    }
    tts_arch arch_type = TTSCPP_SUPPORTED_ARCHITECTURES.at(arch);

    if (params->quantize_type != GGML_TYPE_Q5_0 && params->quantize_type != GGML_TYPE_Q8_0 && params->quantize_type != GGML_TYPE_F16 && params->quantize_type != GGML_TYPE_Q4_0) {
        fprintf(stdout, "Warning, %s is untested for quantization type '%d'. Use at your own risk.\n", arch.c_str(), params->quantize_type);
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    gguf_context_ptr ctx_out { gguf_init_empty() };

    // copy the KV pairs from the input file
    gguf_set_kv(ctx_out.get(), meta_ctx);
    gguf_set_val_u32(ctx_out.get(), "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out.get(), "general.quantization_type", params->quantize_type);
    for (ggml_tensor * tensor = ggml_get_first_tensor(weight_ctx); tensor; tensor = ggml_get_next_tensor(weight_ctx, tensor)) {
        std::string name = ggml_get_name(tensor);
        if (name.size() != 0) {
            gguf_add_tensor(ctx_out.get(), tensor);
        }
    }

    std::vector<do_no_init<uint8_t>> work;

    std::ofstream fout;
    auto close_ofstream = [&]() {
        // Write metadata and close file handler
        if (fout.is_open()) {
            fout.seekp(0);
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_out.get()));
            gguf_get_meta_data(ctx_out.get(), data.data());
            fout.write((const char *) data.data(), data.size());
            fout.close();
        }
    };
    auto new_ofstream = [&]() {
        std::string fname = ofile;
        fout = std::ofstream(fname, std::ios::binary);
        fout.exceptions(std::ofstream::failbit); // fail fast on write errors
        const size_t meta_size = gguf_get_meta_size(ctx_out.get());
        // placeholder for the meta data
        ::zeros(fout, meta_size);
    };
    new_ofstream();
    for (ggml_tensor * cur = ggml_get_first_tensor(weight_ctx); cur; cur = ggml_get_next_tensor(weight_ctx, cur)) {
        enum ggml_type new_type;
        void * new_data;
        size_t new_size;
        std::string name = ggml_get_name(cur);

        if (name.size() == 0) {
            continue;
        }

        if (is_quantizable(arch_type, name, params)) {
            if ((cur->type) != GGML_TYPE_F32) {
                TTS_ABORT("ERROR: All quantized tensors must be transformed from 32bit floats. Tensor, '%s', has improper type, '%d'\n", cur->name, cur->type);
            }
            new_type = params->quantize_type;
            if ((new_type >= GGML_TYPE_IQ2_XXS && new_type <= GGML_TYPE_IQ4_XS)) {
                TTS_ABORT("ERROR: Quantization type '%d' requires an importance matrix.\n", new_type);
            }
            const int64_t nelement_size = ggml_nelements(cur) * 4;
            if (work.size() < (size_t)nelement_size) {
                work.resize(nelement_size); // upper bound on size
            }
            new_data = work.data();
            new_size = quantize_tensor(new_data, cur, nullptr, new_type, params->n_threads);
        } else if ((params->convert_non_quantizable_to_f16 && kokoro_is_f16_compatible(name)) || (params->convert_dac_to_f16 && has_prefix(name, "audio_encoder") && !has_suffix(name, "alpha"))) {
            if ((cur->type) != GGML_TYPE_F32) {
                TTS_ABORT("ERROR: All converted tensors must be transformed from 32bit floats. Tensor, '%s', has improper type, '%d'\n", cur->name, cur->type);
            }
            new_type = GGML_TYPE_F16;
            const int64_t nelement_size = ggml_nelements(cur) * 4;
            if (work.size() < (size_t)nelement_size) {
                work.resize(nelement_size); // upper bound on size
            }
            new_data = work.data();
            new_size = quantize_tensor(new_data, cur, nullptr, new_type, params->n_threads);
        } else {
            new_type = cur->type;
            new_data = cur->data;
            new_size = ggml_nbytes(cur);
        }

        gguf_set_tensor_type(ctx_out.get(), name.c_str(), new_type);
        gguf_set_tensor_data(ctx_out.get(), name.c_str(), new_data);
        fprintf(stdout, "At tensor: '%s' with new size: %zu bytes\n", name.c_str(), new_size);
        // write tensor data + padding
        fout.write((const char *) new_data, new_size);
        zeros(fout, GGML_PAD(new_size, align) - new_size);
    }
    close_ofstream();
}
