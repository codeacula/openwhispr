#include "wav.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "log_mel.h"

namespace whisper_npu {

namespace {

uint16_t u16(const char* p) {
  return static_cast<uint16_t>(static_cast<unsigned char>(p[0]) |
                               (static_cast<unsigned char>(p[1]) << 8));
}

uint32_t u32(const char* p) {
  return static_cast<uint32_t>(u16(p)) | (static_cast<uint32_t>(u16(p + 2)) << 16);
}

constexpr uint16_t kFormatPcm = 1;
constexpr uint16_t kFormatExtensible = 0xFFFE;

}  // namespace

std::vector<float> readPcm16MonoWav(std::string_view data) {
  if (data.size() < 12 || data.substr(0, 4) != "RIFF" || data.substr(8, 4) != "WAVE") {
    throw std::invalid_argument("audio is not a WAV file");
  }

  bool haveFormat = false;
  size_t pos = 12;
  while (pos + 8 <= data.size()) {
    const std::string_view id = data.substr(pos, 4);
    const uint32_t size = u32(data.data() + pos + 4);
    const size_t body = pos + 8;
    const size_t available = data.size() - body;

    if (id == "fmt ") {
      if (size < 16 || available < 16) throw std::invalid_argument("truncated WAV fmt chunk");
      uint16_t format = u16(data.data() + body);
      if (format == kFormatExtensible && size >= 26 && available >= 26) {
        format = u16(data.data() + body + 24);  // first two bytes of the SubFormat GUID
      }
      const uint16_t channels = u16(data.data() + body + 2);
      const uint32_t rate = u32(data.data() + body + 4);
      const uint16_t bits = u16(data.data() + body + 14);
      if (format != kFormatPcm || channels != 1 || rate != kSampleRate || bits != 16) {
        throw std::invalid_argument("expected 16 kHz mono 16-bit PCM, got format " +
                                    std::to_string(format) + ", " + std::to_string(channels) +
                                    " ch, " + std::to_string(rate) + " Hz, " +
                                    std::to_string(bits) + " bit");
      }
      haveFormat = true;
    } else if (id == "data") {
      if (!haveFormat) throw std::invalid_argument("WAV data chunk before fmt chunk");
      // A streamed WAV may carry a placeholder size; trust what was received
      const size_t bytes = std::min<size_t>(size, available) & ~static_cast<size_t>(1);
      std::vector<float> samples(bytes / 2);
      for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = static_cast<int16_t>(u16(data.data() + body + 2 * i)) / 32768.0f;
      }
      return samples;
    }
    pos = body + size + (size & 1);  // chunks are word-aligned
  }
  throw std::invalid_argument("WAV file has no data chunk");
}

}  // namespace whisper_npu
