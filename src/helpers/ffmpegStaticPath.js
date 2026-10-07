const path = require("path");

// ffmpeg-static resolves to null on any platform/arch missing from its own
// list, and win32-arm64 is one of them even with ffmpeg.exe installed beside
// it. ARM64 Windows builds ship the x64 ffmpeg.exe, which runs under the OS's
// x64 emulation, so point at that file instead of giving up.
function resolveFfmpegStaticPath() {
  const resolved = require("ffmpeg-static");
  if (resolved) return resolved;

  if (process.platform === "win32" && process.arch === "arm64") {
    return path.join(path.dirname(require.resolve("ffmpeg-static/package.json")), "ffmpeg.exe");
  }
  return null;
}

module.exports = { resolveFfmpegStaticPath };
