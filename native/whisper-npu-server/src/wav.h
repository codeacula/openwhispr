#pragma once

#include <string_view>
#include <vector>

namespace whisper_npu {

// Samples of a 16 kHz mono PCM16 WAV, scaled to [-1, 1). OpenWhispr converts
// every recording to exactly this format before posting it, so anything else
// is rejected rather than resampled. Throws std::invalid_argument.
std::vector<float> readPcm16MonoWav(std::string_view data);

}  // namespace whisper_npu
