#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace whisper_npu {

// Token ids and decoding for a Hugging Face Whisper tokenizer.json.
class Vocab {
 public:
  explicit Vocab(const std::filesystem::path& tokenizerJson);

  // -1 when the tokenizer has no such token.
  int id(const std::string& token) const;

  // Text of the ordinary tokens; special tokens (id >= endOfText) are skipped.
  std::string decode(const std::vector<int>& ids) const;

  int endOfText() const { return endOfText_; }
  int startOfTranscript() const { return startOfTranscript_; }
  int transcribe() const { return transcribe_; }
  int noTimestamps() const { return noTimestamps_; }
  int noSpeech() const { return noSpeech_; }
  int size() const { return static_cast<int>(bytes_.size()); }

  // Language code -> token id; empty for English-only (.en) models.
  const std::unordered_map<std::string, int>& languages() const { return languages_; }
  std::string languageCode(int tokenId) const;

 private:
  std::vector<std::string> bytes_;  // id -> raw bytes
  std::unordered_map<std::string, int> ids_;
  std::unordered_map<std::string, int> languages_;
  int endOfText_ = -1;
  int startOfTranscript_ = -1;
  int transcribe_ = -1;
  int noTimestamps_ = -1;
  int noSpeech_ = -1;
};

}  // namespace whisper_npu
