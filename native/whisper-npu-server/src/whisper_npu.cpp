#include "whisper_npu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "text_util.h"

namespace whisper_npu {

namespace {

// Qualcomm's reference app masks unused cache slots with -100, not -inf,
// because the model runs in fp16.
constexpr float kMaskBlocked = -100.0f;
// Shortest tail worth decoding once the audio has been split into windows
constexpr size_t kMinWindowSamples = kSampleRate / 10;
// Long audio is cut at the quietest 20 ms in the last 5 s of each window
constexpr size_t kSplitSearchSamples = 5 * kSampleRate;
constexpr size_t kSplitFrameSamples = kSampleRate / 50;

using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

size_t elementSize(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return 4;
    default:
      throw std::runtime_error("unsupported tensor element type " + std::to_string(type));
  }
}

void zeroTensor(Ort::Value& value) {
  const auto info = value.GetTensorTypeAndShapeInfo();
  std::memset(value.GetTensorMutableRawData(), 0,
              info.GetElementCount() * elementSize(info.GetElementType()));
}

std::vector<float> tensorToFloats(const Ort::Value& value) {
  const auto info = value.GetTensorTypeAndShapeInfo();
  const size_t count = info.GetElementCount();
  std::vector<float> out(count);
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    const auto* data = value.GetTensorData<Ort::Float16_t>();
    for (size_t i = 0; i < count; ++i) out[i] = data[i].ToFloat();
  } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const auto* data = value.GetTensorData<float>();
    std::copy(data, data + count, out.begin());
  } else {
    throw std::runtime_error("logits are neither float16 nor float32");
  }
  return out;
}

bool endsWith(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

float logSumExp(const std::vector<float>& scores) {
  const float peak = *std::max_element(scores.begin(), scores.end());
  double sum = 0;
  for (float score : scores) {
    if (score != -std::numeric_limits<float>::infinity()) sum += std::exp(score - peak);
  }
  return peak + static_cast<float>(std::log(sum));
}

// Greedy decoding can lock into a loop ("the the the ..."). When the tail is
// one block repeated, keep a single copy and report it so decoding stops.
bool trimRepetition(std::vector<int>& tokens) {
  for (size_t period = 1; period <= 32; ++period) {
    const size_t repeats = period == 1 ? 10 : period < 4 ? 5 : 3;
    const size_t span = period * repeats;
    if (tokens.size() < span) break;
    const size_t start = tokens.size() - span;
    bool periodic = true;
    for (size_t i = start + period; i < tokens.size() && periodic; ++i) {
      periodic = tokens[i] == tokens[i - period];
    }
    if (periodic) {
      tokens.resize(tokens.size() - period * (repeats - 1));
      return true;
    }
  }
  return false;
}

size_t quietestSplit(const std::vector<float>& samples, size_t start) {
  const size_t searchStart = start + kWindowSamples - kSplitSearchSamples;
  size_t best = start + kWindowSamples;
  double bestEnergy = std::numeric_limits<double>::max();
  for (size_t frame = searchStart; frame + kSplitFrameSamples <= start + kWindowSamples;
       frame += kSplitFrameSamples / 2) {
    double energy = 0;
    for (size_t i = frame; i < frame + kSplitFrameSamples; ++i) energy += samples[i] * samples[i];
    if (energy < bestEnergy) {
      bestEnergy = energy;
      best = frame + kSplitFrameSamples / 2;
    }
  }
  return best - start;
}

}  // namespace

WhisperNpu::WhisperNpu(OrtRuntime& runtime, const std::filesystem::path& modelDir)
    : runtime_(runtime), vocab_(modelDir / L"tokenizer.json") {
  cpu_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  encoder_ = Ort::Session(runtime_.env(), (modelDir / L"encoder.onnx").c_str(),
                          runtime_.npuSessionOptions());
  decoder_ = Ort::Session(runtime_.env(), (modelDir / L"decoder.onnx").c_str(),
                          runtime_.npuSessionOptions());

  Ort::AllocatorWithDefaultOptions allocator;

  encoderInput_ = encoder_.GetInputNameAllocated(0, allocator).get();
  // GetTensorTypeAndShapeInfo() views into its TypeInfo, which must outlive it
  const Ort::TypeInfo encoderType = encoder_.GetInputTypeInfo(0);
  const auto encoderInfo = encoderType.GetTensorTypeAndShapeInfo();
  encoderInputShape_ = encoderInfo.GetShape();
  encoderInputType_ = encoderInfo.GetElementType();
  if (encoderInputShape_.size() != 3 || encoderInputShape_[2] != kMelFrames) {
    throw std::runtime_error("encoder input is not [1, mel bins, 3000]");
  }
  mel_ = std::make_unique<LogMel>(static_cast<int>(encoderInputShape_[1]));
  for (size_t i = 0; i < encoder_.GetOutputCount(); ++i) {
    encoderOutputs_.push_back(encoder_.GetOutputNameAllocated(i, allocator).get());
  }
  crossInputIndex_.assign(encoderOutputs_.size(), -1);

  for (size_t i = 0; i < decoder_.GetInputCount(); ++i) {
    const std::string name = decoder_.GetInputNameAllocated(i, allocator).get();
    const Ort::TypeInfo type = decoder_.GetInputTypeInfo(i);
    const auto info = type.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();
    const int index = static_cast<int>(decoderInputNames_.size());
    decoderInputNames_.push_back(name);

    const auto cross = std::find(encoderOutputs_.begin(), encoderOutputs_.end(), name);
    if (name == "input_ids") {
      inputIdsIndex_ = index;
      decoderInputs_.push_back(Ort::Value::CreateTensor<int32_t>(
          cpu_, inputIds_.data(), inputIds_.size(), shape.data(), shape.size()));
    } else if (name == "position_ids") {
      positionIdsIndex_ = index;
      decoderInputs_.push_back(Ort::Value::CreateTensor<int32_t>(
          cpu_, positionIds_.data(), positionIds_.size(), shape.data(), shape.size()));
    } else if (name == "attention_mask") {
      maskIndex_ = index;
      maskLength_ = static_cast<int>(shape.back());
      mask_.assign(maskLength_, Ort::Float16_t(kMaskBlocked));
      decoderInputs_.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
          cpu_, mask_.data(), mask_.size(), shape.data(), shape.size()));
    } else if (cross != encoderOutputs_.end()) {
      crossInputIndex_[cross - encoderOutputs_.begin()] = index;
      decoderInputs_.emplace_back(nullptr);  // filled by each encoder run
    } else {
      // Self-attention KV cache: starts at zero and is replaced by the matching
      // *_out tensor after every step
      decoderInputs_.push_back(
          Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), info.GetElementType()));
    }
  }
  if (inputIdsIndex_ < 0 || maskIndex_ < 0 || maskLength_ < 8) {
    throw std::runtime_error("decoder does not look like an AI Hub Whisper decoder");
  }
  if (std::find(crossInputIndex_.begin(), crossInputIndex_.end(), -1) != crossInputIndex_.end()) {
    throw std::runtime_error("an encoder output has no matching decoder input");
  }

  for (size_t i = 0; i < decoder_.GetOutputCount(); ++i) {
    const std::string name = decoder_.GetOutputNameAllocated(i, allocator).get();
    decoderOutputNames_.push_back(name);
    int feeds = -1;
    if (name == "logits") {
      logitsOutput_ = static_cast<int>(i);
    } else if (endsWith(name, "_out")) {
      const std::string input = name.substr(0, name.size() - 4) + "_in";
      const auto it = std::find(decoderInputNames_.begin(), decoderInputNames_.end(), input);
      if (it != decoderInputNames_.end()) feeds = static_cast<int>(it - decoderInputNames_.begin());
    }
    outputFeedsInput_.push_back(feeds);
  }
  if (logitsOutput_ < 0) throw std::runtime_error("decoder has no logits output");
}

void WhisperNpu::runEncoder(const float* samples, size_t count) {
  const std::vector<float> mel = mel_->compute(samples, count);
  const char* inputName = encoderInput_.c_str();
  std::vector<const char*> outputNames;
  for (const auto& name : encoderOutputs_) outputNames.push_back(name.c_str());

  std::vector<Ort::Float16_t> half;
  Ort::Value input{nullptr};
  if (encoderInputType_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    half.reserve(mel.size());
    for (float value : mel) half.emplace_back(value);
    input = Ort::Value::CreateTensor<Ort::Float16_t>(cpu_, half.data(), half.size(),
                                                     encoderInputShape_.data(),
                                                     encoderInputShape_.size());
  } else {
    input = Ort::Value::CreateTensor<float>(cpu_, const_cast<float*>(mel.data()), mel.size(),
                                            encoderInputShape_.data(), encoderInputShape_.size());
  }

  auto outputs = encoder_.Run(Ort::RunOptions{nullptr}, &inputName, &input, 1, outputNames.data(),
                              outputNames.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    decoderInputs_[crossInputIndex_[i]] = std::move(outputs[i]);
  }
}

void WhisperNpu::resetDecoder() {
  std::fill(mask_.begin(), mask_.end(), Ort::Float16_t(kMaskBlocked));
  for (size_t i = 0; i < decoderInputs_.size(); ++i) {
    const bool selfCache =
        static_cast<int>(i) != inputIdsIndex_ && static_cast<int>(i) != positionIdsIndex_ &&
        static_cast<int>(i) != maskIndex_ &&
        std::find(crossInputIndex_.begin(), crossInputIndex_.end(), static_cast<int>(i)) ==
            crossInputIndex_.end();
    if (selfCache) zeroTensor(decoderInputs_[i]);
  }
}

const std::vector<float>& WhisperNpu::step(int position, int token) {
  inputIds_[0] = token;
  positionIds_[0] = position;
  mask_[maskLength_ - position - 1] = Ort::Float16_t(0.0f);

  std::vector<const char*> inputNames;
  for (const auto& name : decoderInputNames_) inputNames.push_back(name.c_str());
  std::vector<const char*> outputNames;
  for (const auto& name : decoderOutputNames_) outputNames.push_back(name.c_str());

  auto outputs = decoder_.Run(Ort::RunOptions{nullptr}, inputNames.data(), decoderInputs_.data(),
                              decoderInputs_.size(), outputNames.data(), outputNames.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (outputFeedsInput_[i] >= 0) decoderInputs_[outputFeedsInput_[i]] = std::move(outputs[i]);
  }
  logits_ = tensorToFloats(outputs[logitsOutput_]);
  return logits_;
}

WhisperNpu::Window WhisperNpu::decodeWindow(const float* samples, size_t count,
                                            const std::string& language, TranscribeStats& stats,
                                            const TranscribeOptions& options) {
  Window window;
  const auto encodeStart = Clock::now();
  runEncoder(samples, count);
  resetDecoder();
  stats.encoderSeconds += secondsSince(encodeStart);
  stats.windows++;

  const auto decodeStart = Clock::now();
  int position = 0;
  std::vector<float> scores = step(position++, vocab_.startOfTranscript());
  stats.decoderSteps++;

  // The logits after <|startoftranscript|> carry both language detection and
  // Whisper's no-speech probability.
  float noSpeechProb = 0.0f;
  if (vocab_.noSpeech() >= 0 && vocab_.noSpeech() < static_cast<int>(scores.size())) {
    noSpeechProb = std::exp(scores[vocab_.noSpeech()] - logSumExp(scores));
  }

  std::vector<int> prompt;
  const auto& languages = vocab_.languages();
  if (languages.empty()) {
    window.language = "en";  // English-only model: no language or task tokens
  } else {
    auto chosen = languages.find(language);
    if (chosen == languages.end()) {
      int best = -1;
      for (const auto& [code, id] : languages) {
        if (id < static_cast<int>(scores.size()) && (best < 0 || scores[id] > scores[best])) best = id;
      }
      chosen = languages.find(vocab_.languageCode(best));
    }
    window.language = chosen->first;
    prompt.push_back(chosen->second);
    prompt.push_back(vocab_.transcribe());
  }
  prompt.push_back(vocab_.noTimestamps());
  for (int token : prompt) {
    scores = step(position++, token);
    stats.decoderSteps++;
  }

  const int endOfText = vocab_.endOfText();
  const int blank = vocab_.id("\xC4\xA0");  // byte-level " "
  double sumLogprob = 0;
  while (true) {
    // Special tokens never appear in the text; the first token may be neither
    // end-of-text nor a bare space (Whisper's suppress_blank)
    for (size_t id = endOfText + 1; id < scores.size(); ++id) {
      scores[id] = -std::numeric_limits<float>::infinity();
    }
    if (window.tokens.empty()) {
      scores[endOfText] = -std::numeric_limits<float>::infinity();
      if (blank >= 0) scores[blank] = -std::numeric_limits<float>::infinity();
    }
    const int next = static_cast<int>(std::max_element(scores.begin(), scores.end()) - scores.begin());
    sumLogprob += scores[next] - logSumExp(scores);
    if (next == endOfText) break;
    window.tokens.push_back(next);
    if (trimRepetition(window.tokens) || position >= maskLength_ - 1) break;
    scores = step(position++, next);
    stats.decoderSteps++;
  }
  stats.decoderSeconds += secondsSince(decodeStart);

  const double avgLogprob = sumLogprob / static_cast<double>(window.tokens.size() + 1);
  if (noSpeechProb > options.noSpeechThreshold && avgLogprob < options.logprobThreshold) {
    window.tokens.clear();
  }
  return window;
}

TranscribeResult WhisperNpu::transcribe(const std::vector<float>& samples,
                                        const TranscribeOptions& options) {
  TranscribeResult result;
  std::string language = options.language;
  for (size_t start = 0; start < samples.size();) {
    const size_t remaining = samples.size() - start;
    const size_t length = remaining > kWindowSamples ? quietestSplit(samples, start) : remaining;
    if (length >= kMinWindowSamples || start == 0) {
      Window window = decodeWindow(samples.data() + start, length, language, result.stats, options);
      // Later windows keep the language detected in the first one
      if (result.language.empty()) {
        result.language = window.language;
        language = window.language;
      }
      result.text += vocab_.decode(window.tokens);
    }
    start += length;
  }

  const auto first = result.text.find_first_not_of(" \t\r\n");
  const auto last = result.text.find_last_not_of(" \t\r\n");
  result.text = first == std::string::npos ? "" : result.text.substr(first, last - first + 1);
  return result;
}

}  // namespace whisper_npu
