#include "vocab.h"

#include <windows.h>

#include <algorithm>
#include <array>
#include <climits>
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

// UTF-8 for a code point below U+0800 (byte-level symbols stop at U+0143)
std::string utf8(int codePoint) {
  std::string out;
  if (codePoint < 0x80) {
    out.push_back(static_cast<char>(codePoint));
  } else {
    out.push_back(static_cast<char>(0xC0 | (codePoint >> 6)));
    out.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
  }
  return out;
}

struct CodePoint {
  char32_t value;
  size_t offset;  // byte offset in the source text
};

// Malformed bytes become U+FFFD one byte at a time, so offsets always advance
std::vector<CodePoint> decodeUtf8(const std::string& text) {
  std::vector<CodePoint> out;
  for (size_t i = 0; i < text.size();) {
    const unsigned char lead = static_cast<unsigned char>(text[i]);
    size_t length = lead < 0x80            ? 1
                    : (lead >> 5) == 0x6   ? 2
                    : (lead >> 4) == 0xE   ? 3
                    : (lead >> 3) == 0x1E  ? 4
                                           : 0;
    char32_t value = length == 1   ? lead
                     : length == 2 ? (lead & 0x1F)
                     : length == 3 ? (lead & 0x0F)
                                   : (lead & 0x07);
    for (size_t k = 1; k < length; ++k) {
      const unsigned char next = i + k < text.size() ? static_cast<unsigned char>(text[i + k]) : 0;
      if ((next & 0xC0) != 0x80) {
        length = 0;
        break;
      }
      value = (value << 6) | (next & 0x3F);
    }
    if (length == 0) {
      out.push_back({0xFFFD, i});
      i += 1;
    } else {
      out.push_back({value, i});
      i += length;
    }
  }
  return out;
}

enum class CharClass { Letter, Number, Space, Other };

// Python's \s, which GPT-2's pattern was written against
bool isSpace(char32_t c) {
  return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
         c == 0x205F || c == 0x3000;
}

// \p{L} and \p{N} by the Windows character tables, which agree with them for
// the scripts dictation uses (Latin, Cyrillic, Greek, CJK, ...)
CharClass classify(char32_t c) {
  if (isSpace(c)) return CharClass::Space;
  if (c < 0x80) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return CharClass::Letter;
    if (c >= '0' && c <= '9') return CharClass::Number;
    return CharClass::Other;
  }
  WCHAR units[2] = {};
  int count = 1;
  if (c <= 0xFFFF) {
    units[0] = static_cast<WCHAR>(c);
  } else {
    const char32_t v = c - 0x10000;
    units[0] = static_cast<WCHAR>(0xD800 + (v >> 10));
    units[1] = static_cast<WCHAR>(0xDC00 + (v & 0x3FF));
    count = 2;
  }
  WORD types[2] = {};
  if (!GetStringTypeW(CT_CTYPE1, units, count, types)) return CharClass::Other;
  if (types[0] & C1_ALPHA) return CharClass::Letter;
  if (types[0] & C1_DIGIT) return CharClass::Number;
  return CharClass::Other;
}

// GPT-2's pre-tokenizer pattern, applied by hand since std::regex has no
// Unicode classes:
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
std::vector<std::string> preTokenize(const std::string& text) {
  const auto codePoints = decodeUtf8(text);
  const size_t n = codePoints.size();
  std::vector<CharClass> classes(n);
  for (size_t k = 0; k < n; ++k) classes[k] = classify(codePoints[k].value);

  auto slice = [&](size_t from, size_t to) {
    const size_t end = to < n ? codePoints[to].offset : text.size();
    return text.substr(codePoints[from].offset, end - codePoints[from].offset);
  };
  auto runEnd = [&](size_t from, CharClass cls) {
    while (from < n && classes[from] == cls) ++from;
    return from;
  };
  auto asciiAt = [&](size_t k, char c) {
    return k < n && codePoints[k].value == static_cast<char32_t>(c);
  };

  std::vector<std::string> words;
  for (size_t i = 0; i < n;) {
    if (codePoints[i].value == U'\'') {
      static constexpr const char* kSuffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
      size_t matched = 0;
      for (const char* suffix : kSuffixes) {
        size_t length = 0;
        while (suffix[length] && asciiAt(i + 1 + length, suffix[length])) ++length;
        if (suffix[length] == '\0') {
          matched = length;
          break;
        }
      }
      if (matched > 0) {
        words.push_back(slice(i, i + 1 + matched));
        i += 1 + matched;
        continue;
      }
    }
    // A single ASCII space joins the letter, number or symbol run after it
    const size_t body =
        codePoints[i].value == U' ' && i + 1 < n && classes[i + 1] != CharClass::Space ? i + 1
                                                                                      : i;
    if (classes[body] != CharClass::Space) {
      const size_t end = runEnd(body, classes[body]);
      words.push_back(slice(i, end));
      i = end;
      continue;
    }
    // \s+(?!\S): a whitespace run leaves its last character to the next word
    size_t end = runEnd(i, CharClass::Space);
    if (end < n && end - i > 1) --end;
    words.push_back(slice(i, end));
    i = end;
  }
  return words;
}

}  // namespace

Vocab::Vocab(const std::filesystem::path& tokenizerJson) {
  std::ifstream file(tokenizerJson, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open " + pathToUtf8(tokenizerJson));
  const auto json = nlohmann::json::parse(file);

  const auto table = codePointToByte();
  for (int codePoint = 0; codePoint < static_cast<int>(table.size()); ++codePoint) {
    if (table[codePoint] >= 0) byteSymbols_[table[codePoint]] = utf8(codePoint);
  }
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

  // Merges are "left right" strings, or [left, right] pairs in newer exports
  int rank = 0;
  for (const auto& merge : json.at("model").at("merges")) {
    mergeRanks_.emplace(merge.is_string() ? merge.get<std::string>()
                                          : merge.at(0).get<std::string>() + " " +
                                                merge.at(1).get<std::string>(),
                        rank++);
  }

  endOfText_ = id("<|endoftext|>");
  startOfTranscript_ = id("<|startoftranscript|>");
  transcribe_ = id("<|transcribe|>");
  noTimestamps_ = id("<|notimestamps|>");
  noSpeech_ = id("<|nospeech|>");
  if (noSpeech_ < 0) noSpeech_ = id("<|nocaptions|>");  // pre-v3 name
  startOfPrev_ = id("<|startofprev|>");
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

std::vector<int> Vocab::encode(const std::string& text) const {
  std::vector<int> ids;
  for (const auto& word : preTokenize(text)) {
    const auto wordIds = encodeWord(word);
    ids.insert(ids.end(), wordIds.begin(), wordIds.end());
  }
  return ids;
}

std::vector<int> Vocab::encodeWord(const std::string& word) const {
  std::vector<std::string> symbols;
  for (unsigned char byte : word) symbols.push_back(byteSymbols_[byte]);

  // Byte-pair encoding: merge the adjacent pair with the best rank until no
  // pair of the current symbols has a merge
  while (symbols.size() > 1) {
    int bestRank = INT_MAX;
    size_t best = 0;
    for (size_t k = 0; k + 1 < symbols.size(); ++k) {
      const auto it = mergeRanks_.find(symbols[k] + " " + symbols[k + 1]);
      if (it != mergeRanks_.end() && it->second < bestRank) {
        bestRank = it->second;
        best = k;
      }
    }
    if (bestRank == INT_MAX) break;
    const std::string left = symbols[best];
    const std::string right = symbols[best + 1];
    std::vector<std::string> merged;
    merged.reserve(symbols.size());
    for (size_t k = 0; k < symbols.size(); ++k) {
      if (k + 1 < symbols.size() && symbols[k] == left && symbols[k + 1] == right) {
        merged.push_back(left + right);
        ++k;
      } else {
        merged.push_back(symbols[k]);
      }
    }
    symbols.swap(merged);
  }

  std::vector<int> ids;
  for (const auto& symbol : symbols) {
    const int tokenId = id(symbol);
    if (tokenId >= 0) {
      ids.push_back(tokenId);
      continue;
    }
    // A consistent tokenizer has every merge result in its vocabulary; if
    // this one does not, fall back to the symbol's single-byte tokens
    for (const auto& codePoint : decodeUtf8(symbol)) {
      const size_t length = codePoint.value < 0x80 ? 1 : 2;
      const int byteId = id(symbol.substr(codePoint.offset, length));
      if (byteId >= 0) ids.push_back(byteId);
    }
  }
  return ids;
}

std::string Vocab::languageCode(int tokenId) const {
  for (const auto& [code, id] : languages_) {
    if (id == tokenId) return code;
  }
  return {};
}

}  // namespace whisper_npu
