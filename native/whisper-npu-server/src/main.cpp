// whisper-npu-server: Whisper on the Snapdragon Hexagon NPU behind the subset
// of whisper.cpp's whisper-server HTTP API that OpenWhispr uses, so the app
// drives it exactly like whisper-server (src/helpers/whisperServer.js).
//
//   GET  /           health check, answers once the model is loaded
//   POST /inference  multipart: file (16 kHz mono PCM16 WAV), language, prompt,
//                    logprob_thold, no_speech_thold -> {"text", "language"}
//
// Arguments whisper-server takes but this server has no use for (--threads,
// --max-len, --vad*, ...) are accepted and ignored.
//
// `whisper-npu-server --probe` lists the NPU devices the QNN provider sees as
// JSON and exits 0 when there is one, 2 when there is none.
//
// `whisper-npu-server --tokenize <tokenizer.json>` prints the token ids of the
// UTF-8 text on stdin as JSON, for checking the encoder against Hugging Face.

#include "httplib.h"

#include <windows.h>

#include <fcntl.h>
#include <io.h>

#include <chrono>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <filesystem>
#include <mutex>
#include <string>

#include "json.hpp"
#include "ort_runtime.h"
#include "text_util.h"
#include "vocab.h"
#include "wav.h"
#include "whisper_npu.h"

namespace {

using whisper_npu::pathToUtf8;

constexpr int kExitNoNpu = 2;
constexpr int kExitRuntime = 3;
constexpr int kExitModel = 4;
constexpr int kExitListen = 5;
constexpr int kExitUsage = 64;

struct Args {
  std::filesystem::path model;
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string language = "auto";
  bool probe = false;
  std::filesystem::path tokenizer;  // --tokenize
};

std::string wideToUtf8(const wchar_t* text) {
  const int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
  std::string out(size > 0 ? size - 1 : 0, '\0');
  if (size > 1) WideCharToMultiByte(CP_UTF8, 0, text, -1, out.data(), size, nullptr, nullptr);
  return out;
}

bool parseArgs(int argc, wchar_t** argv, Args& args) {
  for (int i = 1; i < argc; ++i) {
    const std::string flag = wideToUtf8(argv[i]);
    const bool hasValue = i + 1 < argc && std::wstring(argv[i + 1]).rfind(L"--", 0) != 0;
    if (flag == "--probe") {
      args.probe = true;
    } else if (flag == "--tokenize" && hasValue) {
      args.tokenizer = argv[++i];
    } else if ((flag == "--model" || flag == "-m") && hasValue) {
      args.model = argv[++i];
    } else if (flag == "--host" && hasValue) {
      args.host = wideToUtf8(argv[++i]);
    } else if (flag == "--port" && hasValue) {
      args.port = std::stoi(wideToUtf8(argv[++i]));
    } else if ((flag == "--language" || flag == "-l") && hasValue) {
      args.language = wideToUtf8(argv[++i]);
    } else if (hasValue) {
      ++i;  // a whisper-server option this server ignores, with its value
    }
  }
  return args.probe || !args.model.empty() || !args.tokenizer.empty();
}

std::filesystem::path executableDir() {
  std::wstring path(MAX_PATH, L'\0');
  while (true) {
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (length < path.size()) {
      path.resize(length);
      break;
    }
    path.resize(path.size() * 2);
  }
  return std::filesystem::path(path).parent_path();
}

// A multibyte character split across two 30 s windows decodes to invalid
// UTF-8; replace it rather than let dump() throw and fail the request.
std::string toJson(const nlohmann::json& value) {
  return value.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

const char* deviceTypeName(OrtHardwareDeviceType type) {
  switch (type) {
    case OrtHardwareDeviceType_CPU:
      return "cpu";
    case OrtHardwareDeviceType_GPU:
      return "gpu";
    case OrtHardwareDeviceType_NPU:
      return "npu";
    default:
      return "unknown";
  }
}

// Every execution-provider device ONNX Runtime sees, so a machine without a
// usable NPU can be told apart from one where the QNN provider failed to load.
int probe(whisper_npu::OrtRuntime& runtime) {
  nlohmann::json devices = nlohmann::json::array();
  for (const auto& device : runtime.env().GetEpDevices()) {
    devices.push_back({
        {"ep", device.EpName()},
        {"type", deviceTypeName(device.Device().Type())},
        {"vendor", device.Device().Vendor()},
        {"deviceMetadata", device.Device().Metadata().GetKeyValuePairs()},
        {"epMetadata", device.EpMetadata().GetKeyValuePairs()},
    });
  }
  const bool npu = !runtime.npuDevices().empty();
  std::printf("%s\n", toJson({{"npu", npu}, {"devices", devices}}).c_str());
  return npu ? 0 : kExitNoNpu;
}

int tokenize(const std::filesystem::path& tokenizerJson) {
  std::cin.sync_with_stdio(false);
  std::string text((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
  const whisper_npu::Vocab vocab(tokenizerJson);
  std::printf("%s\n", toJson(vocab.encode(text)).c_str());
  return 0;
}

std::string formValue(const httplib::Request& req, const char* name, const std::string& fallback) {
  return req.has_file(name) ? req.get_file_value(name).content : fallback;
}

float formFloat(const httplib::Request& req, const char* name, float fallback) {
  try {
    return req.has_file(name) ? std::stof(req.get_file_value(name).content) : fallback;
  } catch (...) {
    return fallback;
  }
}

void replyError(httplib::Response& res, int status, const std::string& message) {
  res.status = status;
  res.set_content(toJson({{"error", message}}), "application/json");
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  Args args;
  try {
    if (!parseArgs(argc, argv, args)) {
      std::fprintf(stderr, "usage: whisper-npu-server --model <dir> [--host H] [--port P] "
                           "[--language L] | --probe\n");
      return kExitUsage;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "whisper-npu-server: bad argument: %s\n", e.what());
    return kExitUsage;
  }

  if (!args.tokenizer.empty()) {
    try {
      _setmode(_fileno(stdin), _O_BINARY);
      return tokenize(args.tokenizer);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "whisper-npu-server: %s\n", e.what());
      return kExitModel;
    }
  }

  std::unique_ptr<whisper_npu::OrtRuntime> runtime;
  try {
    runtime = std::make_unique<whisper_npu::OrtRuntime>(executableDir());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "whisper-npu-server: runtime error: %s\n", e.what());
    return kExitRuntime;
  }
  if (args.probe) return probe(*runtime);

  std::unique_ptr<whisper_npu::WhisperNpu> model;
  try {
    const auto start = std::chrono::steady_clock::now();
    model = std::make_unique<whisper_npu::WhisperNpu>(*runtime, args.model);
    std::fprintf(stderr, "whisper-npu-server: loaded %s in %.2fs\n", pathToUtf8(args.model).c_str(),
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "whisper-npu-server: cannot load model %s: %s\n",
                 pathToUtf8(args.model).c_str(), e.what());
    return runtime->npuDevices().empty() ? kExitNoNpu : kExitModel;
  }

  std::mutex inferenceLock;
  httplib::Server server;

  server.Get("/", [](const httplib::Request&, httplib::Response& res) {
    res.set_content(R"({"status":"ok","backend":"qnn-npu"})", "application/json");
  });

  server.Post("/inference", [&](const httplib::Request& req, httplib::Response& res) {
    if (!req.has_file("file")) return replyError(res, 400, "missing 'file' field");

    std::vector<float> samples;
    try {
      samples = whisper_npu::readPcm16MonoWav(req.get_file_value("file").content);
    } catch (const std::exception& e) {
      return replyError(res, 400, e.what());
    }

    whisper_npu::TranscribeOptions options;
    options.language = formValue(req, "language", args.language);
    options.prompt = formValue(req, "prompt", "");
    options.logprobThreshold = formFloat(req, "logprob_thold", options.logprobThreshold);
    options.noSpeechThreshold = formFloat(req, "no_speech_thold", options.noSpeechThreshold);

    std::lock_guard<std::mutex> guard(inferenceLock);
    try {
      const auto start = std::chrono::steady_clock::now();
      const auto result = model->transcribe(samples, options);
      const double elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      std::fprintf(stderr,
                   "whisper-npu-server: %.1fs of audio in %.2fs (encoder %.2fs, decoder %.2fs, "
                   "%d steps, %d window(s), language %s, %d prompt tokens, %d prompt retries)\n",
                   samples.size() / static_cast<double>(whisper_npu::kSampleRate), elapsed,
                   result.stats.encoderSeconds, result.stats.decoderSeconds,
                   result.stats.decoderSteps, result.stats.windows, result.language.c_str(),
                   result.stats.promptTokens, result.stats.promptRetries);
      res.set_content(toJson({{"text", result.text}, {"language", result.language}}),
                      "application/json");
    } catch (const std::exception& e) {
      std::fprintf(stderr, "whisper-npu-server: transcription failed: %s\n", e.what());
      replyError(res, 500, e.what());
    }
  });

  std::fprintf(stderr, "whisper-npu-server: listening on %s:%d\n", args.host.c_str(), args.port);
  if (!server.listen(args.host, args.port)) {
    std::fprintf(stderr, "whisper-npu-server: cannot listen on %s:%d\n", args.host.c_str(),
                 args.port);
    return kExitListen;
  }
  return 0;
}
