#include "lemon/backends/kokoro_server.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/backend_manager.h"
#include "lemon/runtime_config.h"
#include "lemon/utils/custom_args.h"
#include "lemon/utils/process_manager.h"
#include "lemon/utils/json_utils.h"
#include "lemon/error_types.h"
#include <httplib.h>
#include <iostream>
#include <set>
#include <vector>
#include <lemon/utils/aixlog.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace lemon::utils;

namespace lemon {
namespace backends {

InstallParams KokoroServer::get_install_params(const std::string& backend, const std::string& version) {
    InstallParams params;

    if (backend == "hip") {
        // No prebuilt HIP kokoro-hip-server is published - users supply one
        // via LEMONADE_KOKORO_HIP_BIN (built by lemondate's in-tree
        // src/kokoro-hip-server/). Returning empty repo/filename is fine
        // because install_from_github short-circuits as soon as
        // find_external_backend_binary() returns a path.
        params.repo = "";
        params.filename = "";
        return params;
    }

    params.repo = "lemonade-sdk/Kokoros";

#ifdef _WIN32
    params.filename = "kokoros-windows-x86_64.tar.gz";
#elif defined(__linux__)
    params.filename = "kokoros-linux-x86_64.tar.gz";
#else
    throw std::runtime_error("Unsupported platform for kokoros");
#endif

    return params;
}

KokoroServer::KokoroServer(const std::string& log_level, ModelManager* model_manager, BackendManager* backend_manager)
    : WrappedServer("kokoro-server", log_level, model_manager, backend_manager) {

}

KokoroServer::~KokoroServer() {
    unload();
}

void KokoroServer::load(const std::string& model_name, const ModelInfo& model_info, const RecipeOptions& options, bool do_not_upgrade) {
    LOG(INFO, "KokoroServer") << "Loading model: " << model_name << std::endl;
    LOG(INFO, "KokoroServer") << "Per-model settings: " << options.to_log_string() << std::endl;

    // Resolve backend. Read the raw config so we can distinguish "auto"/unset from
    // an explicit user choice. The recipe_options dynamic-default fallback would
    // otherwise always return the first supported backend (cpu) which hides the
    // LEMONADE_KOKORO_HIP_BIN auto-detect heuristic below.
    auto* cfg = RuntimeConfig::global();
    std::string kokoro_backend = options.get_option("kokoro_backend");
    std::string raw_backend = cfg ? cfg->backend_string("kokoro", "backend") : "";
    if (raw_backend.empty() || raw_backend == "auto") {
        std::string hip_bin = cfg ? cfg->backend_string("kokoro", "hip_bin") : "";
        if (!hip_bin.empty() && hip_bin != "builtin" && fs::exists(hip_bin)) {
            kokoro_backend = "hip";
        } else {
            kokoro_backend = "cpu";
        }
    }

    RuntimeConfig::validate_backend_choice("kokoro", kokoro_backend);

    LOG(INFO, "KokoroServer") << "Using backend: " << kokoro_backend << std::endl;

    if (kokoro_backend == "hip") {
        device_type_ = DEVICE_GPU;
        load_hip(model_name, model_info, options, do_not_upgrade);
        return;
    }

    device_type_ = DEVICE_CPU;

    // Install kokoros if needed
    backend_manager_->install_backend(SPEC.recipe, "cpu");

    // Use pre-resolved model path
    fs::path model_path = fs::path(model_info.resolved_path());
    if (model_path.empty() || !fs::exists(model_path)) {
        throw std::runtime_error("Model file not found for checkpoint: " + model_info.checkpoint());
    }

    json model_index;

    try {
        LOG(INFO, "KokoroServer") << "Reading " << model_path.filename() << std::endl;
        model_index = JsonUtils::load_from_file(model_path.string());
    } catch (const std::exception& e) {
        throw std::runtime_error("Warning: Could not load " + model_path.filename().string() + ": " + e.what());
    }

    LOG(INFO, "KokoroServer") << "Using model: " << model_index["model"] << std::endl;

    // Get koko executable path
    std::string exe_path = BackendUtils::get_backend_binary_path(SPEC, "cpu");

    // Choose a port
    port_ = choose_port();
    if (port_ == 0) {
        throw std::runtime_error("Failed to find an available port");
    }

    LOG(INFO, "KokoroServer") << "Starting server on port " << port_ << std::endl;

    std::vector<std::pair<std::string, std::string>> env_vars;
    fs::path exe_dir = fs::path(exe_path).parent_path();
    env_vars.push_back({"ESPEAK_DATA_PATH", exe_dir.string() + "espeak-ng-data"});
#ifndef _WIN32
    std::string lib_path = exe_dir.string();
    // Preserve existing LD_LIBRARY_PATH if it exists
    const char* existing_ld_path = std::getenv("LD_LIBRARY_PATH");
    if (existing_ld_path && strlen(existing_ld_path) > 0) {
        lib_path = lib_path + ":" + std::string(existing_ld_path);
    }

    env_vars.push_back({"LD_LIBRARY_PATH", lib_path});
    LOG(INFO, "KokoroServer") << "Setting LD_LIBRARY_PATH=" << lib_path << std::endl;
#endif

    // Build command line arguments
    // Note: Don't include exe_path here - ProcessManager::start_process already handles it
    fs::path model_dir = model_path.parent_path();
    std::vector<std::string> args = {
        "-m", (model_dir / model_index["model"]).string(),
        "-d", (model_dir / model_index["voices"]).string(),
        "openai",
        "--ip", "127.0.0.1",
        "--port", std::to_string(port_)
    };

    // Launch the subprocess
    process_handle_ = utils::ProcessManager::start_process(
        exe_path,
        args,
        "",     // working_dir (empty = current)
        is_debug(),  // inherit_output
        false,
        env_vars
    );

    if (process_handle_.pid == 0) {
        throw std::runtime_error("Failed to start koko process");
    }

    LOG(INFO, "KokoroServer") << "Process started with PID: " << process_handle_.pid << std::endl;

    // Wait for server to be ready
    if (!wait_for_ready("/")) {
        unload();
        throw std::runtime_error("koko failed to start or become ready");
    }
}

void KokoroServer::load_hip(const std::string& model_name,
                            const ModelInfo& model_info,
                            const RecipeOptions& options,
                            bool do_not_upgrade) {
    // Short-circuits the GitHub download path - get_install_params("hip") returns
    // empty repo/filename so install_from_github exits as soon as
    // find_external_backend_binary() returns the LEMONADE_KOKORO_HIP_BIN path.
    backend_manager_->install_backend(SPEC.recipe, "hip");

    // Resolve kokoro-hip-server.exe path. This reads config["kokoro"]["hip_bin"]
    // which is populated from LEMONADE_KOKORO_HIP_BIN by config_file.cpp.
    std::string exe_path = BackendUtils::get_backend_binary_path(SPEC, "hip");

    // Resolve the .gguf model path. Precedence:
    //   1) LEMONADE_KOKORO_HIP_MODEL / config["kokoro"]["hip_model"]
    //   2) model_info.resolved_path() if it points at a .gguf file
    auto* cfg = RuntimeConfig::global();
    std::string model_path = cfg ? cfg->backend_string("kokoro", "hip_model") : "";
    if (model_path.empty()) {
        std::string resolved = model_info.resolved_path();
        if (!resolved.empty() && fs::path(resolved).extension() == ".gguf") {
            model_path = resolved;
        }
    }
    if (model_path.empty() || !fs::exists(model_path)) {
        throw std::runtime_error(
            "kokoro-hip-server model not found. Set LEMONADE_KOKORO_HIP_MODEL "
            "(or kokoro.hip_model in config.json) to a .gguf file path.");
    }

    LOG(INFO, "KokoroServer") << "Using model: " << model_path << std::endl;
    LOG(INFO, "KokoroServer") << "Using executable: " << exe_path << std::endl;

    // Choose a port
    port_ = choose_port();
    if (port_ == 0) {
        throw std::runtime_error("Failed to find an available port");
    }

    LOG(INFO, "KokoroServer") << "Starting kokoro-hip-server on port " << port_ << std::endl;

    // Build command line arguments. Lemonade manages the model path, host, and port;
    // optional kokoro-hip-server flags (e.g. --voice) come from kokoro_args.
    // Note: Don't include exe_path here - ProcessManager::start_process already handles it.
    std::vector<std::string> args = {
        "--host", "127.0.0.1",
        "--port", std::to_string(port_),
        "--model", model_path
    };

    std::set<std::string> reserved_flags = {
        "--host",
        "--port",
        "--model"
    };

    std::string kokoro_args = options.get_option("kokoro_args");
    if (!kokoro_args.empty()) {
        std::string validation_error = validate_custom_args(kokoro_args, reserved_flags);
        if (!validation_error.empty()) {
            throw std::invalid_argument(
                "Invalid custom kokoro-hip-server arguments:\n" + validation_error
            );
        }

        LOG(DEBUG, "KokoroServer") << "Adding custom arguments: " << kokoro_args << std::endl;
        std::vector<std::string> custom_args_vec = parse_custom_args(kokoro_args);
        args.insert(args.end(), custom_args_vec.begin(), custom_args_vec.end());
    }

    // Set up environment variables for shared library loading (mirrors whisper_server)
    std::vector<std::pair<std::string, std::string>> env_vars;
    fs::path exe_dir = fs::path(exe_path).parent_path();

#ifndef _WIN32
    std::string lib_path = exe_dir.string();
    const char* existing_ld_path = std::getenv("LD_LIBRARY_PATH");
    if (existing_ld_path && strlen(existing_ld_path) > 0) {
        lib_path = lib_path + ":" + std::string(existing_ld_path);
    }
    env_vars.push_back({"LD_LIBRARY_PATH", lib_path});
    if (is_debug()) {
        std::cout << "[KokoroServer] Setting LD_LIBRARY_PATH=" << lib_path << std::endl;
    }
#endif

    // Launch the subprocess
    process_handle_ = utils::ProcessManager::start_process(
        exe_path,
        args,
        "",     // working_dir (empty = current)
        is_debug(),  // inherit_output
        false,  // filter_health_logs
        env_vars
    );

    if (process_handle_.pid == 0) {
        throw std::runtime_error("Failed to start kokoro-hip-server process");
    }

    LOG(INFO, "KokoroServer") << "Process started with PID: " << process_handle_.pid << std::endl;

    // Wait for server to be ready. kokoro-hip-server responds to GET /health
    // with HTTP 200 once model loading and HIP init are complete.
    if (!wait_for_ready("/health")) {
        unload();
        throw std::runtime_error("kokoro-hip-server failed to start or become ready");
    }

    LOG(INFO, "KokoroServer") << "Server is ready!" << std::endl;
}

void KokoroServer::unload() {
    if (process_handle_.pid != 0) {
        LOG(INFO, "KokoroServer") << "Stopping server (PID: " << process_handle_.pid << ")" << std::endl;
        utils::ProcessManager::stop_process(process_handle_);
        port_ = 0;
        process_handle_ = {nullptr, 0};
    }
}

// ICompletionServer implementation (not supported - return errors)
json KokoroServer::chat_completion(const json& request) {
    return json{
        {"error", {
            {"message", "Kokoro does not support text completion. Use audio speech endpoints instead."},
            {"type", "unsupported_operation"},
            {"code", "model_not_applicable"}
        }}
    };
}

json KokoroServer::completion(const json& request) {
    return json{
        {"error", {
            {"message", "Kokoro does not support text completion. Use audio speech endpoints instead."},
            {"type", "unsupported_operation"},
            {"code", "model_not_applicable"}
        }}
    };
}

json KokoroServer::responses(const json& request) {
    return json{
        {"error", {
            {"message", "Kokoro does not support text completion. Use audio speech endpoints instead."},
            {"type", "unsupported_operation"},
            {"code", "model_not_applicable"}
        }}
    };
}

void KokoroServer::audio_speech(const json& request, httplib::DataSink& sink) {
    json tts_request = request;
    tts_request["model"] = "kokoro";

    // OpenAI does not define "stream" for the speech endpoint
    // relying solely on stream_format. Kokoros requires this boolean
    if (request.contains("stream_format")) {
        tts_request["stream"] = true;
    }

    forward_streaming_request("/v1/audio/speech", tts_request.dump(), sink, false);
}

} // namespace backends
} // namespace lemon
