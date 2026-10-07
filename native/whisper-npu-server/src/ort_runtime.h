#pragma once

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <memory>
#include <vector>

namespace whisper_npu {

// The bundled ONNX Runtime plus Qualcomm's QNN execution provider plugin.
//
// onnxruntime.dll is loaded by full path rather than imported: Windows 11
// ships an older onnxruntime.dll in System32, and some loader configurations
// resolve the bare name to it (OpenWhispr #2054). The QNN plugin does not
// import onnxruntime.dll, so this is the only place it is loaded.
class OrtRuntime {
 public:
  explicit OrtRuntime(const std::filesystem::path& runtimeDir);

  Ort::Env& env() { return *env_; }

  // NPU devices the QNN provider exposes; empty on machines without one.
  std::vector<Ort::ConstEpDevice> npuDevices() const;

  // Session options that run every node on the NPU. Precompiled QNN models
  // are EPContext graphs, so a CPU fallback could only fail later and slower.
  Ort::SessionOptions npuSessionOptions();

 private:
  std::unique_ptr<Ort::Env> env_;
};

}  // namespace whisper_npu
