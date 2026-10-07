#pragma once

#include <array>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace whisper_npu {

// Token ids, encoding and decoding for a Hugging Face Whisper tokenizer.json
// (GPT-2 byte-level BPE).
class Vocab {
 public:
  explicit Vocab(const std::filesystem::path& tokenizerJson);

  // -1 when the tokenizer has no such token.
  int id(const std::string& token) const;

  // Text of the ordinary tokens; special tokens (id >= endOfText) are skipped.
  std::string decode(const std::vector<int>& ids) const;

  // Ordinary tokens for UTF-8 text, split by GPT-2's pre-tokenizer pattern and
  // merged by the tokenizer's BPE ranks. Special-token text is not recognized.
  std::vector<int> encode(const std::string& text) const;

  int endOfText() const { return endOfText_; }
  int startOfTranscript() const { return startOfTranscript_; }
  int transcribe() const { return transcribe_; }
  int noTimestamps() const { return noTimestamps_; }
  int noSpeech() const { return noSpeech_; }
  int startOfPrev() const { return startOfPrev_; }
  int size() const { return static_cast<int>(bytes_.size()); }

  // Language code -> token id; empty for English-only (.en) models.
  const std::unordered_map<std::string, int>& languages() const { return languages_; }
  std::string languageCode(int tokenId) const;

 private:
  std::vector<int> encodeWord(const std::string& word) const;

  std::vector<std::string> bytes_;  // id -> raw bytes
  std::unordered_map<std::string, int> ids_;
  // "left right" (the tokenizer's merge spelling) -> merge priority
  std::unordered_map<std::string, int> mergeRanks_;
  // Raw byte -> its byte-level code point as UTF-8
  std::array<std::string, 256> byteSymbols_;
  std::unordered_map<std::string, int> languages_;
  int endOfText_ = -1;
  int startOfTranscript_ = -1;
  int transcribe_ = -1;
  int noTimestamps_ = -1;
  int noSpeech_ = -1;
  int startOfPrev_ = -1;
};

}  // namespace whisper_npu
