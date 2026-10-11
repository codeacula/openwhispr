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
// Most cache slots a prompt may take; a third of the cache at most
constexpr int kMaxPromptTokens = 64;
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
// Spans are not monotonic (period 3 needs 15 tokens, period 4 only 12), so a
// period that does not fit yet must not stop the longer ones being checked.
bool trimRepetition(std::vector<int>& tokens) {
  for (size_t period = 1; period <= 32; ++period) {
    const size_t repeats = period == 1 ? 10 : period < 4 ? 5 : 3;
    const size_t span = period * repeats;
    if (tokens.size() < span) continue;
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

void WhisperNpu::encodeWindow(const float* samples, size_t count, TranscribeStats& stats) {
  const auto start = Clock::now();
  runEncoder(samples, count);
  stats.encoderSeconds += secondsSince(start);
  stats.windows++;
}

std::string WhisperNpu::pickLanguage(const std::vector<float>& scores) const {
  int best = -1;
  for (const auto& [code, id] : vocab_.languages()) {
    if (id < static_cast<int>(scores.size()) && (best < 0 || scores[id] > scores[best])) best = id;
  }
  return vocab_.languageCode(best);
}

WhisperNpu::Window WhisperNpu::decodeWindow(const std::string& language,
                                            const std::vector<int>& prompt,
                                            TranscribeStats& stats,
                                            const TranscribeOptions& options) {
  Window window;
  const auto decodeStart = Clock::now();
  const int startOfTranscript = vocab_.startOfTranscript();
  const auto& languages = vocab_.languages();
  int position = 0;
  std::vector<float> scores;
  float noSpeechProb = 0.0f;
  auto feed = [&](int token) {
    scores = step(position++, token);
    stats.decoderSteps++;
    // The logits after <|startoftranscript|> carry Whisper's no-speech probability
    if (token == startOfTranscript && vocab_.noSpeech() >= 0 &&
        vocab_.noSpeech() < static_cast<int>(scores.size())) {
      noSpeechProb = std::exp(scores[vocab_.noSpeech()] - logSumExp(scores));
    }
  };

  // Whisper detects the language from <|startoftranscript|> alone, so a
  // prompt in another language cannot sway it; without a prompt that is the
  // first step of the decode itself
  std::string chosen = languages.empty() ? "en" : language;
  const bool detect = !languages.empty() && !languages.count(chosen);
  if (detect && !prompt.empty()) {
    resetDecoder();
    feed(startOfTranscript);
    chosen = pickLanguage(scores);
    position = 0;
  }

  // [<|startofprev|> prompt...] <|startoftranscript|> [<|lang|> <|transcribe|>] <|notimestamps|>
  resetDecoder();
  if (!prompt.empty()) {
    feed(vocab_.startOfPrev());
    for (int token : prompt) feed(token);
  }
  feed(startOfTranscript);
  if (detect && prompt.empty()) chosen = pickLanguage(scores);
  window.language = chosen;
  if (!languages.empty()) {
    feed(languages.at(chosen));
    feed(vocab_.transcribe());
  }
  feed(vocab_.noTimestamps());

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
    const int next =
        static_cast<int>(std::max_element(scores.begin(), scores.end()) - scores.begin());
    sumLogprob += scores[next] - logSumExp(scores);
    if (next == endOfText) break;
    window.tokens.push_back(next);
    if (trimRepetition(window.tokens)) break;
    if (position >= maskLength_ - 1) {
      window.ranOutOfCache = true;
      break;
    }
    feed(next);
  }
  stats.decoderSeconds += secondsSince(decodeStart);

  const double avgLogprob = sumLogprob / static_cast<double>(window.tokens.size() + 1);
  if (noSpeechProb > options.noSpeechThreshold && avgLogprob < options.logprobThreshold) {
    window.tokens.clear();
  }
  return window;
}

std::vector<int> WhisperNpu::promptTokens(const std::string& prompt) const {
  const auto first = prompt.find_first_not_of(" \t\r\n");
  if (first == std::string::npos || vocab_.startOfPrev() < 0) return {};
  const auto last = prompt.find_last_not_of(" \t\r\n");
  // Encoded as given, as whisper.cpp does (OpenAI's Python Whisper adds a
  // leading space), so the NPU reads the dictionary the way the CPU path does
  std::vector<int> tokens = vocab_.encode(prompt.substr(first, last - first + 1));
  // Every prompt token is a self-attention cache slot the transcript cannot
  // use; keep the end, as Whisper does when a prompt is too long
  const size_t limit = static_cast<size_t>(std::min(kMaxPromptTokens, (maskLength_ - 1) / 3));
  if (tokens.size() > limit) tokens.erase(tokens.begin(), tokens.end() - limit);
  return tokens;
}

TranscribeResult WhisperNpu::transcribe(const std::vector<float>& samples,
                                        const TranscribeOptions& options) {
  TranscribeResult result;
  const std::vector<int> prompt = promptTokens(options.prompt);
  result.stats.promptTokens = static_cast<int>(prompt.size());
  std::string language = options.language;
  for (size_t start = 0; start < samples.size();) {
    const size_t remaining = samples.size() - start;
    const size_t length = remaining > kWindowSamples ? quietestSplit(samples, start) : remaining;
    if (length >= kMinWindowSamples || start == 0) {
      encodeWindow(samples.data() + start, length, result.stats);
      Window window = decodeWindow(language, prompt, result.stats, options);
      // The prompt took cache slots the transcript needed: decode the window
      // again without it rather than lose its last words
      if (window.ranOutOfCache && !prompt.empty()) {
        result.stats.promptRetries++;
        window = decodeWindow(language, {}, result.stats, options);
      }
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
