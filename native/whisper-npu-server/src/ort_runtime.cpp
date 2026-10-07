#include "ort_runtime.h"

#include <windows.h>

#include <stdexcept>
#include <string>

#include "text_util.h"

namespace whisper_npu {

namespace {

// The QNN plugin names its devices' provider after the registration name
constexpr const char* kQnnEpName = "QNNExecutionProvider";

const OrtApi* loadOrtApi(const std::filesystem::path& runtimeDir) {
  const std::filesystem::path dll = runtimeDir / L"onnxruntime.dll";
  HMODULE module = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!module) {
    throw std::runtime_error("cannot load " + pathToUtf8(dll) + " (Windows error " +
                             std::to_string(GetLastError()) + ")");
  }

  using GetApiBaseFn = const OrtApiBase*(ORT_API_CALL*)();
  auto getApiBase = reinterpret_cast<GetApiBaseFn>(GetProcAddress(module, "OrtGetApiBase"));
  if (!getApiBase) {
    throw std::runtime_error(pathToUtf8(dll) + " does not export OrtGetApiBase");
  }

  const OrtApiBase* base = getApiBase();
  const OrtApi* api = base->GetApi(ORT_API_VERSION);
  if (!api) {
    throw std::runtime_error("onnxruntime " + std::string(base->GetVersionString()) +
                             " does not provide ORT API version " +
                             std::to_string(ORT_API_VERSION));
  }
  return api;
}

}  // namespace

OrtRuntime::OrtRuntime(const std::filesystem::path& runtimeDir) {
  Ort::InitApi(loadOrtApi(runtimeDir));
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "whisper-npu");
  env_->RegisterExecutionProviderLibrary(
      kQnnEpName, (runtimeDir / L"onnxruntime_providers_qnn.dll").wstring());
}

std::vector<Ort::ConstEpDevice> OrtRuntime::npuDevices() const {
  std::vector<Ort::ConstEpDevice> npus;
  for (const auto& device : env_->GetEpDevices()) {
    if (std::string(device.EpName()) == kQnnEpName &&
        device.Device().Type() == OrtHardwareDeviceType_NPU) {
      npus.push_back(device);
    }
  }
  return npus;
}

Ort::SessionOptions OrtRuntime::npuSessionOptions() {
  const auto devices = npuDevices();
  if (devices.empty()) {
    throw std::runtime_error("no Qualcomm NPU found by the QNN execution provider");
  }
  Ort::SessionOptions options;
  options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
  options.AppendExecutionProvider_V2(*env_, {devices.front()},
                                     std::unordered_map<std::string, std::string>{});
  return options;
}

}  // namespace whisper_npu
