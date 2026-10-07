#pragma once

#include <cstddef>
#include <vector>

namespace whisper_npu {

constexpr int kSampleRate = 16000;
constexpr int kWindowSamples = 30 * kSampleRate;  // Whisper's fixed 30 s window
constexpr int kMelFrames = 3000;

// Whisper's log-mel spectrogram: 25 ms Hann window, 10 ms hop, Slaney mel
// filters, log10 clamped to 8 dB below the peak, scaled to roughly [-1, 1].
class LogMel {
 public:
  explicit LogMel(int melBins);

  int melBins() const { return melBins_; }

  // Up to 30 s of 16 kHz mono audio; shorter input is zero-padded.
  // Returns melBins x 3000, row-major.
  std::vector<float> compute(const float* samples, size_t count) const;

 private:
  int melBins_;
  std::vector<float> filters_;  // melBins x kFreqBins
  std::vector<float> window_;
  std::vector<float> cos_;  // kFreqBins x kFftSize
  std::vector<float> sin_;
};

}  // namespace whisper_npu
