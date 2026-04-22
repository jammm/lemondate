// ============================================================
// kokoro-hip-server / main.cpp
//
// Minimal cpp-httplib server that wraps ttscpp's Kokoro
// inference. Exposes an OpenAI-compatible /v1/audio/speech
// endpoint + a /health probe. lemond launches this binary via
// its `KokoroServer` backend (kokoros_backend=hip) as a drop-in
// replacement for the koboldcpp HIP dll.
//
// The behaviour contract (request/response shape) mirrors what
// koboldcpp serves at /v1/audio/speech today:
//
//   POST /v1/audio/speech
//   Content-Type: application/json
//   Body: {"model":"kokoro","input":"<text>","voice":"af_bella",
//          "response_format":"wav"}
//   → 200 audio/wav (16-bit PCM, 24 kHz mono)
//   → 400 application/json {"error": "..."} on bad request
//   → 500 application/json {"error": "..."} on inference failure
//
// Inference is a single-shot ttscpp call guarded by a mutex so
// concurrent requests serialize on the runner (ttscpp's runner
// holds a single ggml_backend_sched that isn't re-entrant).
// ============================================================

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "ttscpp.h"
#include "ttscommon.h"

using json = nlohmann::json;

// ------------------------------------------------------------
// Crash diagnostics. ttscpp's TTS_ABORT/TTS_ASSERT macros call
// std::abort(), which on Windows ends up as __fastfail(7) =
// FAST_FAIL_FATAL_APP_EXIT and gets logged as
// "Exception Code c0000409 subcode 7" in WER but without the
// ttscpp abort message surviving to where we can read it (lemond
// doesn't pipe the subprocess stderr to its own log reliably).
//
// g_last_request is a best-effort last-known-input snapshot, and
// install_crash_handlers() traps SIGABRT / std::terminate /
// unhandled exceptions so we can dump the text + a marker to a
// file *before* the CRT fast-fails us out.
// ------------------------------------------------------------
static std::mutex g_last_mu;
static std::string g_last_text;
static std::string g_last_voice;
static std::string g_crash_log_path;

static void write_crash_record(const char * reason) {
    std::string snap_text, snap_voice;
    {
        std::lock_guard<std::mutex> lk(g_last_mu);
        snap_text = g_last_text;
        snap_voice = g_last_voice;
    }
    fprintf(stderr,
        "\n[kokoro-hip-server CRASH] %s\n"
        "  last_voice=%s\n"
        "  last_text_len=%zu\n"
        "  last_text=%s\n",
        reason, snap_voice.c_str(), snap_text.size(), snap_text.c_str());
    fflush(stderr);

    if (!g_crash_log_path.empty()) {
        if (FILE * f = fopen(g_crash_log_path.c_str(), "a")) {
            std::time_t t = std::time(nullptr);
            char ts[64];
            std::strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
            fprintf(f,
                "[%s] %s\n"
                "  voice=%s\n"
                "  text_len=%zu\n"
                "  text=%s\n"
                "  text_hex=",
                ts, reason, snap_voice.c_str(), snap_text.size(), snap_text.c_str());
            for (unsigned char c : snap_text) fprintf(f, "%02x", (unsigned)c);
            fprintf(f, "\n\n");
            fclose(f);
        }
    }
}

static void crash_sigabrt(int) {
    write_crash_record("SIGABRT");
    std::signal(SIGABRT, SIG_DFL);
    std::raise(SIGABRT);
}

static void crash_sigsegv(int) {
    write_crash_record("SIGSEGV");
    std::signal(SIGSEGV, SIG_DFL);
    std::raise(SIGSEGV);
}

static void crash_terminate() {
    const char * reason = "std::terminate";
    std::string msg = reason;
    try {
        if (auto eptr = std::current_exception()) {
            std::rethrow_exception(eptr);
        }
    } catch (const std::exception & e) {
        msg = std::string("std::terminate (what=") + e.what() + ")";
    } catch (...) {
        msg = "std::terminate (unknown exception)";
    }
    write_crash_record(msg.c_str());
    std::abort();
}

static void install_crash_handlers() {
    const char * env = std::getenv("KOKORO_HIP_CRASH_LOG");
    if (env && *env) {
        g_crash_log_path = env;
    } else {
        // default next to the executable's working dir
        g_crash_log_path = "kokoro-hip-server.crash.log";
    }
    std::signal(SIGABRT, crash_sigabrt);
    std::signal(SIGSEGV, crash_sigsegv);
    std::set_terminate(crash_terminate);
    fprintf(stderr, "[kokoro-hip-server] crash log: %s\n", g_crash_log_path.c_str());
}

// ------------------------------------------------------------
// CLI options (simple custom parser — CLI11 is overkill for 4
// flags and would needlessly pull in another dependency on
// top of the one src/lemond already uses).
// ------------------------------------------------------------
struct cli_opts {
    std::string host = "127.0.0.1";
    int port = 5001;
    std::string model;
    std::string voice = "af_bella";
    bool cpu_only = false;
    int n_threads = 0; // 0 -> auto (hardware concurrency)
};

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "Usage: %s --model <path> [--host HOST] [--port N] [--voice NAME]\n"
        "                       [--cpu-only] [--threads N]\n"
        "\n"
        "Required:\n"
        "  --model PATH      Path to a Kokoro .gguf model file\n"
        "                    (e.g. Kokoro_no_espeak_Q4.gguf)\n"
        "\n"
        "Optional:\n"
        "  --host HOST       Bind address (default: 127.0.0.1)\n"
        "  --port N          TCP port (default: 5001)\n"
        "  --voice NAME      Default Kokoro voice (default: af_bella)\n"
        "  --cpu-only        Disable GPU backend (default: off -> use HIP)\n"
        "  --threads N       CPU threads for inference (default: auto)\n"
        "  --help            Show this help and exit\n",
        argv0);
}

static bool parse_cli(int argc, char ** argv, cli_opts & out) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_val = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s expects a value\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--model" || a == "-m") {
            const char * v = need_val("--model");
            if (!v) return false;
            out.model = v;
        } else if (a == "--host") {
            const char * v = need_val("--host");
            if (!v) return false;
            out.host = v;
        } else if (a == "--port" || a == "-p") {
            const char * v = need_val("--port");
            if (!v) return false;
            try {
                out.port = std::stoi(v);
            } catch (...) {
                fprintf(stderr, "error: --port must be an integer, got %s\n", v);
                return false;
            }
        } else if (a == "--voice" || a == "-v") {
            const char * v = need_val("--voice");
            if (!v) return false;
            out.voice = v;
        } else if (a == "--cpu-only") {
            out.cpu_only = true;
        } else if (a == "--threads" || a == "-t") {
            const char * v = need_val("--threads");
            if (!v) return false;
            try {
                out.n_threads = std::stoi(v);
            } catch (...) {
                fprintf(stderr, "error: --threads must be an integer, got %s\n", v);
                return false;
            }
        } else {
            fprintf(stderr, "error: unknown arg '%s'\n", a.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    if (out.model.empty()) {
        fprintf(stderr, "error: --model is required\n");
        print_usage(argv[0]);
        return false;
    }
    if (out.n_threads <= 0) {
        out.n_threads = std::max((int) std::thread::hardware_concurrency(), 1);
    }
    return true;
}

// ------------------------------------------------------------
// WAV (RIFF/WAVE, PCM 16-bit mono) encoder.
//
// The Kokoro runner emits float samples in [-1, 1] (approx) at
// 24 000 Hz. We clip-and-quantize to int16 and wrap in a
// standard canonical WAVE container so any OAI client (ffmpeg,
// browser <audio>, speak.py) can play it back directly.
// ------------------------------------------------------------
namespace wav {

static inline void write_u16_le(std::vector<uint8_t> & buf, uint16_t v) {
    buf.push_back((uint8_t)(v & 0xff));
    buf.push_back((uint8_t)((v >> 8) & 0xff));
}

static inline void write_u32_le(std::vector<uint8_t> & buf, uint32_t v) {
    buf.push_back((uint8_t)(v & 0xff));
    buf.push_back((uint8_t)((v >> 8) & 0xff));
    buf.push_back((uint8_t)((v >> 16) & 0xff));
    buf.push_back((uint8_t)((v >> 24) & 0xff));
}

static inline void write_tag(std::vector<uint8_t> & buf, const char tag[4]) {
    buf.insert(buf.end(), tag, tag + 4);
}

// Encode float PCM samples -> int16 PCM WAV bytes.
// sample_rate is in Hz (24000 for Kokoro).
static std::vector<uint8_t> encode_pcm16(const float * samples,
                                         size_t n_samples,
                                         uint32_t sample_rate) {
    constexpr uint16_t num_channels = 1;
    constexpr uint16_t bits_per_sample = 16;
    const uint16_t block_align = num_channels * (bits_per_sample / 8);
    const uint32_t byte_rate = sample_rate * block_align;
    const uint32_t data_size = (uint32_t)(n_samples * block_align);

    std::vector<uint8_t> out;
    out.reserve(44 + data_size);

    // RIFF header
    write_tag(out, "RIFF");
    write_u32_le(out, 36 + data_size); // chunk size
    write_tag(out, "WAVE");

    // fmt sub-chunk
    write_tag(out, "fmt ");
    write_u32_le(out, 16);              // PCM fmt chunk size
    write_u16_le(out, 1);               // audio format = PCM
    write_u16_le(out, num_channels);
    write_u32_le(out, sample_rate);
    write_u32_le(out, byte_rate);
    write_u16_le(out, block_align);
    write_u16_le(out, bits_per_sample);

    // data sub-chunk
    write_tag(out, "data");
    write_u32_le(out, data_size);

    out.resize(44 + data_size);
    uint8_t * dst = out.data() + 44;
    for (size_t i = 0; i < n_samples; ++i) {
        float s = samples[i];
        if (s > 1.0f) s = 1.0f;
        else if (s < -1.0f) s = -1.0f;
        int32_t q = (int32_t) std::lrint(s * 32767.0f);
        if (q > 32767) q = 32767;
        else if (q < -32768) q = -32768;
        uint16_t u = (uint16_t) (int16_t) q;
        dst[2 * i + 0] = (uint8_t) (u & 0xff);
        dst[2 * i + 1] = (uint8_t) ((u >> 8) & 0xff);
    }
    return out;
}

} // namespace wav

// ------------------------------------------------------------
// Model load + thread-safe inference wrapper.
// ------------------------------------------------------------
struct kokoro_service {
    tts_runner * runner = nullptr;
    std::string  default_voice;
    std::mutex   inference_mu;
    uint32_t     sample_rate = 24000;

    bool load(const std::string & model_path,
              const std::string & default_voice_,
              int n_threads,
              bool cpu_only) {
        default_voice = default_voice_;
        generation_configuration config(default_voice);
        runner = runner_from_file(model_path, n_threads, &config, cpu_only);
        if (!runner) {
            fprintf(stderr, "[kokoro-hip-server] failed to load model: %s\n",
                    model_path.c_str());
            return false;
        }
        if (runner->arch != KOKORO_ARCH) {
            fprintf(stderr,
                "[kokoro-hip-server] model at %s is not a Kokoro model "
                "(got arch=%d). This server only handles Kokoro.\n",
                model_path.c_str(), (int) runner->arch);
            return false;
        }
        sample_rate = (uint32_t) runner->sampling_rate;
        fprintf(stderr, "[kokoro-hip-server] loaded %s (arch=kokoro, sr=%u Hz, default_voice=%s)\n",
                model_path.c_str(), sample_rate, default_voice.c_str());
        return true;
    }

    // Returns 0 on success. Populates `wav_out`. On failure, fills `err`.
    int synthesize(const std::string & text,
                   const std::string & voice,
                   std::vector<uint8_t> & wav_out,
                   std::string & err) {
        if (!runner) {
            err = "model not loaded";
            return 1;
        }
        std::lock_guard<std::mutex> lock(inference_mu);

        // Snapshot for the crash handlers. TTS_ABORT / TTS_ASSERT inside
        // ttscpp can invoke std::abort() which on Windows becomes a
        // fast-fail with no surviving stderr. If we crash mid-generate,
        // the crash log tells us exactly what text + voice tripped it.
        {
            std::lock_guard<std::mutex> lk(g_last_mu);
            g_last_text = text;
            g_last_voice = voice.empty() ? default_voice : voice;
        }

        // Re-point the runner's config to the request's voice.
        generation_configuration config(voice.empty() ? default_voice : voice);

        tts_response resp{};
        int rc = ::generate(runner, text, &resp, &config);
        if (rc != 0 || resp.n_outputs == 0 || resp.data == nullptr) {
            err = "inference failed (rc=" + std::to_string(rc) +
                  ", n_outputs=" + std::to_string(resp.n_outputs) + ")";
            return rc == 0 ? 1 : rc;
        }
        wav_out = wav::encode_pcm16(resp.data, resp.n_outputs, sample_rate);
        return 0;
    }
};

// ------------------------------------------------------------
// HTTP glue.
// ------------------------------------------------------------
static void send_json_error(httplib::Response & res, int status, const std::string & msg) {
    json j;
    j["error"] = {{"message", msg}, {"type", "tts_error"}};
    res.status = status;
    res.set_content(j.dump(), "application/json");
    fprintf(stderr, "[kokoro-hip-server] %d %s\n", status, msg.c_str());
}

static void handle_speech(kokoro_service & svc, const httplib::Request & req, httplib::Response & res) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception & e) {
        send_json_error(res, 400, std::string("invalid JSON body: ") + e.what());
        return;
    }
    if (!body.is_object()) {
        send_json_error(res, 400, "request body must be a JSON object");
        return;
    }

    std::string text;
    if (body.contains("input") && body["input"].is_string()) {
        text = body["input"].get<std::string>();
    } else if (body.contains("text") && body["text"].is_string()) {
        text = body["text"].get<std::string>();
    }
    if (text.empty()) {
        send_json_error(res, 400, "missing or empty 'input' (or 'text') field");
        return;
    }

    std::string voice;
    if (body.contains("voice") && body["voice"].is_string()) {
        voice = body["voice"].get<std::string>();
    }

    // response_format is advisory — we always return PCM16 WAV because
    // that is what ttscpp produces natively at 24 kHz. If a client
    // requests something else we note it but still send wav (same as
    // koboldcpp's behaviour).
    std::string response_format = "wav";
    if (body.contains("response_format") && body["response_format"].is_string()) {
        response_format = body["response_format"].get<std::string>();
        if (response_format != "wav" && response_format != "pcm") {
            fprintf(stderr,
                "[kokoro-hip-server] warn: response_format='%s' not supported, sending wav\n",
                response_format.c_str());
        }
    }

    std::vector<uint8_t> wav_bytes;
    std::string err;
    int rc = svc.synthesize(text, voice, wav_bytes, err);
    if (rc != 0) {
        send_json_error(res, 500, err.empty() ? "inference failed" : err);
        return;
    }

    res.status = 200;
    res.set_header("Content-Disposition", "attachment; filename=\"speech.wav\"");
    res.set_content(std::string(wav_bytes.begin(), wav_bytes.end()), "audio/wav");
}

int main(int argc, char ** argv) {
    install_crash_handlers();

    cli_opts opts;
    if (!parse_cli(argc, argv, opts)) {
        return 2;
    }

    // Sanity-check the model file up front — ttscpp's error path is
    // an abort() on a malformed gguf, so better to fail fast here
    // with a clear message than to die inside ggml.
    {
        std::ifstream f(opts.model, std::ios::binary);
        if (!f.good()) {
            fprintf(stderr, "[kokoro-hip-server] model file not found: %s\n",
                    opts.model.c_str());
            return 3;
        }
    }

    kokoro_service svc;
    if (!svc.load(opts.model, opts.voice, opts.n_threads, opts.cpu_only)) {
        return 4;
    }

    httplib::Server server;

    server.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.status = 200;
        res.set_content("ok", "text/plain");
    });

    server.Post("/v1/audio/speech", [&svc](const httplib::Request & req, httplib::Response & res) {
        handle_speech(svc, req, res);
    });

    // koboldcpp also accepts /api/extra/tts and /audio/speech for the same
    // payload; we mirror that so lemond's `kokoros_backend=hip` path can be
    // a drop-in for the current /v1/audio/speech router.
    server.Post("/audio/speech", [&svc](const httplib::Request & req, httplib::Response & res) {
        handle_speech(svc, req, res);
    });
    server.Post("/api/extra/tts", [&svc](const httplib::Request & req, httplib::Response & res) {
        handle_speech(svc, req, res);
    });

    server.set_logger([](const httplib::Request & req, const httplib::Response & res) {
        fprintf(stderr, "[kokoro-hip-server] %s %s -> %d\n",
                req.method.c_str(), req.path.c_str(), res.status);
    });

    fprintf(stderr, "[kokoro-hip-server] listening on http://%s:%d\n",
            opts.host.c_str(), opts.port);
    fflush(stderr);

    if (!server.listen(opts.host, opts.port)) {
        fprintf(stderr, "[kokoro-hip-server] failed to bind %s:%d\n",
                opts.host.c_str(), opts.port);
        return 5;
    }
    return 0;
}
