#include "log_mel.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <thread>

namespace whisper_npu {

namespace {

constexpr int kFftSize = 400;
constexpr int kHop = 160;
constexpr int kFreqBins = kFftSize / 2 + 1;

// Slaney mel scale, matching librosa.filters.mel(htk=False), which is what
// Whisper's bundled filter banks were generated with.
double hzToMel(double hz) {
  constexpr double kMinLogHz = 1000.0;
  constexpr double kMinLogMel = 15.0;
  const double logStep = std::log(6.4) / 27.0;
  return hz >= kMinLogHz ? kMinLogMel + std::log(hz / kMinLogHz) / logStep : hz / (200.0 / 3);
}

double melToHz(double mel) {
  constexpr double kMinLogMel = 15.0;
  const double logStep = std::log(6.4) / 27.0;
  return mel >= kMinLogMel ? 1000.0 * std::exp(logStep * (mel - kMinLogMel)) : mel * (200.0 / 3);
}

std::vector<float> slaneyFilters(int melBins) {
  std::vector<double> melPoints(melBins + 2);
  const double maxMel = hzToMel(kSampleRate / 2.0);
  for (int i = 0; i < melBins + 2; ++i) {
    melPoints[i] = melToHz(maxMel * i / (melBins + 1));
  }

  std::vector<float> filters(static_cast<size_t>(melBins) * kFreqBins, 0.0f);
  for (int m = 0; m < melBins; ++m) {
    const double lower = melPoints[m];
    const double center = melPoints[m + 1];
    const double upper = melPoints[m + 2];
    const double norm = 2.0 / (upper - lower);
    for (int k = 0; k < kFreqBins; ++k) {
      const double freq = static_cast<double>(k) * kSampleRate / kFftSize;
      const double rising = (freq - lower) / (center - lower);
      const double falling = (upper - freq) / (upper - center);
      const double weight = std::max(0.0, std::min(rising, falling));
      filters[static_cast<size_t>(m) * kFreqBins + k] = static_cast<float>(weight * norm);
    }
  }
  return filters;
}

}  // namespace

LogMel::LogMel(int melBins) : melBins_(melBins), filters_(slaneyFilters(melBins)) {
  window_.resize(kFftSize);
  for (int n = 0; n < kFftSize; ++n) {
    // Periodic Hann, as torch.hann_window(400) uses
    window_[n] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * n / kFftSize));
  }
  cos_.resize(static_cast<size_t>(kFreqBins) * kFftSize);
  sin_.resize(static_cast<size_t>(kFreqBins) * kFftSize);
  for (int k = 0; k < kFreqBins; ++k) {
    for (int n = 0; n < kFftSize; ++n) {
      const double angle = 2.0 * std::numbers::pi * k * n / kFftSize;
      cos_[static_cast<size_t>(k) * kFftSize + n] = static_cast<float>(std::cos(angle));
      sin_[static_cast<size_t>(k) * kFftSize + n] = static_cast<float>(std::sin(angle));
    }
  }
}

std::vector<float> LogMel::compute(const float* samples, size_t count) const {
  // Zero-pad to 30 s, then reflect-pad by half a frame on each side (STFT
  // center=True); the STFT yields 3001 frames and Whisper drops the last.
  constexpr int kPad = kFftSize / 2;
  std::vector<float> padded(kWindowSamples + 2 * kPad, 0.0f);
  const size_t used = std::min<size_t>(count, kWindowSamples);
  std::copy(samples, samples + used, padded.begin() + kPad);
  for (int i = 0; i < kPad; ++i) {
    padded[kPad - 1 - i] = padded[kPad + 1 + i];
    padded[kPad + kWindowSamples + i] = padded[kPad + kWindowSamples - 2 - i];
  }

  // Power spectrum by direct DFT: 400 is not a power of two, and at
  // 3000 x 201 x 400 the cost is a few tens of ms spread over the cores.
  std::vector<float> power(static_cast<size_t>(kMelFrames) * kFreqBins);
  auto worker = [&](int firstFrame, int lastFrame) {
    std::vector<float> frame(kFftSize);
    for (int f = firstFrame; f < lastFrame; ++f) {
      const float* src = padded.data() + static_cast<size_t>(f) * kHop;
      for (int n = 0; n < kFftSize; ++n) frame[n] = src[n] * window_[n];
      for (int k = 0; k < kFreqBins; ++k) {
        const float* c = cos_.data() + static_cast<size_t>(k) * kFftSize;
        const float* s = sin_.data() + static_cast<size_t>(k) * kFftSize;
        float re = 0.0f;
        float im = 0.0f;
        for (int n = 0; n < kFftSize; ++n) {
          re += frame[n] * c[n];
          im += frame[n] * s[n];
        }
        power[static_cast<size_t>(f) * kFreqBins + k] = re * re + im * im;
      }
    }
  };
  const int threads = static_cast<int>(std::clamp(std::thread::hardware_concurrency(), 1u, 8u));
  std::vector<std::thread> pool;
  const int perThread = (kMelFrames + threads - 1) / threads;
  for (int t = 0; t < threads; ++t) {
    const int first = t * perThread;
    const int last = std::min(kMelFrames, first + perThread);
    if (first < last) pool.emplace_back(worker, first, last);
  }
  for (auto& thread : pool) thread.join();

  std::vector<float> mel(static_cast<size_t>(melBins_) * kMelFrames);
  float peak = -INFINITY;
  for (int m = 0; m < melBins_; ++m) {
    const float* filter = filters_.data() + static_cast<size_t>(m) * kFreqBins;
    for (int f = 0; f < kMelFrames; ++f) {
      const float* spectrum = power.data() + static_cast<size_t>(f) * kFreqBins;
      float energy = 0.0f;
      for (int k = 0; k < kFreqBins; ++k) energy += filter[k] * spectrum[k];
      const float value = std::log10(std::max(energy, 1e-10f));
      mel[static_cast<size_t>(m) * kMelFrames + f] = value;
      peak = std::max(peak, value);
    }
  }
  for (float& value : mel) value = (std::max(value, peak - 8.0f) + 4.0f) / 4.0f;
  return mel;
}

}  // namespace whisper_npu
