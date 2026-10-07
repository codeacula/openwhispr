const WHISPER_CPP_TAG = process.env.WHISPER_CPP_VERSION || "0.0.10";

const WINDOWS_MSVC_RUNTIME_LIBRARIES = Object.freeze([
  "msvcp140.dll",
  "vcruntime140.dll",
  "vcruntime140_1.dll",
  "vcomp140.dll",
]);

// ggml only builds for Windows ARM64 with Clang, and Clang's OpenMP runtime
// (libomp140.aarch64.dll) ships under VS's debug_nonredist, so the arm64
// build sets GGML_OPENMP=OFF and needs no OpenMP DLL (no vcomp140). No
// vcruntime140_1 either: its __CxxFrameHandler4 is x64-only, and the copy in
// VS's arm64 CRT redist folder is an ARM64EC image (PE header x64) that
// nothing ARM64 imports.
const WINDOWS_ARM64_MSVC_RUNTIME_LIBRARIES = Object.freeze(["msvcp140.dll", "vcruntime140.dll"]);

module.exports = {
  WHISPER_CPP_TAG,
  WINDOWS_MSVC_RUNTIME_LIBRARIES,
  WINDOWS_ARM64_MSVC_RUNTIME_LIBRARIES,
};
