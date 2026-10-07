#include "vocab.h"

#include <array>
#include <fstream>
#include <regex>
#include <stdexcept>

#include "json.hpp"
#include "text_util.h"

namespace whisper_npu {

namespace {

// Inverse of GPT-2's bytes_to_unicode(): byte-level BPE stores every byte as
// a printable code point, so token strings must be mapped back to raw bytes.
std::array<int, 512> codePointToByte() {
  std::array<int, 512> table;
  table.fill(-1);
  std::array<bool, 256> printable{};
  for (int b = '!'; b <= '~'; ++b) printable[b] = true;
  for (int b = 0xA1; b <= 0xAC; ++b) printable[b] = true;
  for (int b = 0xAE; b <= 0xFF; ++b) printable[b] = true;
  int shifted = 0;
  for (int b = 0; b < 256; ++b) {
    if (printable[b]) {
      table[b] = b;
    } else {
      table[256 + shifted++] = b;
    }
  }
  return table;
}

std::string tokenBytes(const std::string& token, const std::array<int, 512>& table) {
  std::string out;
  for (size_t i = 0; i < token.size();) {
    const unsigned char lead = static_cast<unsigned char>(token[i]);
    int codePoint;
    size_t length;
    if (lead < 0x80) {
      codePoint = lead;
      length = 1;
    } else if ((lead & 0xE0) == 0xC0 && i + 1 < token.size()) {
      codePoint = ((lead & 0x1F) << 6) | (token[i + 1] & 0x3F);
      length = 2;
    } else {
      // Byte-level tokens only use code points below U+0200; anything else is
      // a tokenizer this decoder does not understand, so keep it verbatim.
      return token;
    }
    const int byte = codePoint < 512 ? table[codePoint] : -1;
    if (byte < 0) return token;
    out.push_back(static_cast<char>(byte));
    i += length;
  }
  return out;
}

}  // namespace

Vocab::Vocab(const std::filesystem::path& tokenizerJson) {
  std::ifstream file(tokenizerJson, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open " + pathToUtf8(tokenizerJson));
  const auto json = nlohmann::json::parse(file);

  const auto table = codePointToByte();
  int maxId = -1;
  for (const auto& [token, id] : json.at("model").at("vocab").items()) {
    maxId = std::max(maxId, id.get<int>());
  }
  for (const auto& added : json.at("added_tokens")) {
    maxId = std::max(maxId, added.at("id").get<int>());
  }
  bytes_.resize(static_cast<size_t>(maxId) + 1);

  for (const auto& [token, id] : json.at("model").at("vocab").items()) {
    bytes_[id.get<int>()] = tokenBytes(token, table);
    ids_[token] = id.get<int>();
  }
  // Language tokens look like <|en|>, <|yue|>; other specials are longer words
  const std::regex languageToken(R"(<\|([a-z]{2,3})\|>)");
  for (const auto& added : json.at("added_tokens")) {
    const std::string content = added.at("content").get<std::string>();
    const int id = added.at("id").get<int>();
    bytes_[id] = content;
    ids_[content] = id;
    std::smatch match;
    if (std::regex_match(content, match, languageToken)) languages_[match[1]] = id;
  }

  endOfText_ = id("<|endoftext|>");
  startOfTranscript_ = id("<|startoftranscript|>");
  transcribe_ = id("<|transcribe|>");
  noTimestamps_ = id("<|notimestamps|>");
  noSpeech_ = id("<|nospeech|>");
  if (noSpeech_ < 0) noSpeech_ = id("<|nocaptions|>");  // pre-v3 name
  if (endOfText_ < 0 || startOfTranscript_ < 0 || noTimestamps_ < 0) {
    throw std::runtime_error(pathToUtf8(tokenizerJson) + " is not a Whisper tokenizer");
  }
}

int Vocab::id(const std::string& token) const {
  const auto it = ids_.find(token);
  return it == ids_.end() ? -1 : it->second;
}

std::string Vocab::decode(const std::vector<int>& ids) const {
  std::string text;
  for (int id : ids) {
    if (id >= 0 && id < endOfText_) text += bytes_[id];
  }
  return text;
}

std::string Vocab::languageCode(int tokenId) const {
  for (const auto& [code, id] : languages_) {
    if (id == tokenId) return code;
  }
  return {};
}

}  // namespace whisper_npu
