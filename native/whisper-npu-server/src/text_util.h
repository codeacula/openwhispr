#pragma once

#include <filesystem>
#include <string>

namespace whisper_npu {

// std::filesystem::path::string() throws on Windows for names outside the
// active code page; error messages and logs always want UTF-8.
inline std::string pathToUtf8(const std::filesystem::path& path) {
  const auto u8 = path.u8string();
  return std::string(u8.begin(), u8.end());
}

inline std::filesystem::path utf8ToPath(const std::string& text) {
  return std::filesystem::path(std::u8string(text.begin(), text.end()));
}

}  // namespace whisper_npu
