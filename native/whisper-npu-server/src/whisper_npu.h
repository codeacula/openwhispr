#pragma once

#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "log_mel.h"
#include "ort_runtime.h"
#include "vocab.h"

namespace whisper_npu {

struct TranscribeOptions {
  // ISO code ("en", "de", ...) or "auto" to detect it from the first window
  std::string language = "auto";
  // Whisper's no-speech rule: a window is silent when P(<|nospeech|>) exceeds
  // noSpeechThreshold and the average token log-probability is below
  // logprobThreshold.
  float noSpeechThreshold = 0.6f;
  float logprobThreshold = -1.0f;
};

struct TranscribeStats {
  double encoderSeconds = 0;
  double decoderSeconds = 0;
  int decoderSteps = 0;
  int windows = 0;
};

struct TranscribeResult {
  std::string text;
  std::string language;
  TranscribeStats stats;
};

// Whisper encoder/decoder pair precompiled for the Hexagon NPU (Qualcomm AI
// Hub "precompiled_qnn_onnx" export): a fixed 30 s encoder and a one-token
// decoder step with a fixed-size self-attention KV cache.
//
// A model directory holds encoder.onnx/.bin, decoder.onnx/.bin and the
// matching Hugging Face tokenizer.json.
class WhisperNpu {
 public:
  WhisperNpu(OrtRuntime& runtime, const std::filesystem::path& modelDir);

  // Not thread-safe: one transcription at a time.
  TranscribeResult transcribe(const std::vector<float>& samples, const TranscribeOptions& options);

 private:
  struct Window {
    std::vector<int> tokens;
    std::string language;
  };

  Window decodeWindow(const float* samples, size_t count, const std::string& language,
                      TranscribeStats& stats, const TranscribeOptions& options);
  void runEncoder(const float* samples, size_t count);
  void resetDecoder();
  const std::vector<float>& step(int position, int token);

  OrtRuntime& runtime_;
  Vocab vocab_;
  Ort::Session encoder_{nullptr};
  Ort::Session decoder_{nullptr};
  std::unique_ptr<LogMel> mel_;
  Ort::MemoryInfo cpu_{nullptr};

  std::string encoderInput_;
  std::vector<int64_t> encoderInputShape_;
  ONNXTensorElementDataType encoderInputType_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  std::vector<std::string> encoderOutputs_;

  // Decoder inputs in session order; values are replaced in place each step
  std::vector<std::string> decoderInputNames_;
  std::vector<Ort::Value> decoderInputs_;
  std::vector<std::string> decoderOutputNames_;
  // For each decoder output: index of the input it feeds next step, or -1
  std::vector<int> outputFeedsInput_;
  int logitsOutput_ = -1;
  int inputIdsIndex_ = -1;
  int positionIdsIndex_ = -1;
  int maskIndex_ = -1;
  // decoder input index for each encoder output (cross-attention KV)
  std::vector<int> crossInputIndex_;

  int maskLength_ = 0;
  std::vector<int32_t> inputIds_{0};
  std::vector<int32_t> positionIds_{0};
  std::vector<Ort::Float16_t> mask_;
  std::vector<float> logits_;
};

}  // namespace whisper_npu
