const { execFile } = require("child_process");
const path = require("path");
const { promisify } = require("util");
const debugLogger = require("./debugLogger");

const CACHE_TTL_MS = 5000;

// ps-list throws on win32-arm64 rather than run its vendored fastlist, but
// the x64 fastlist runs fine under Windows on Arm's x64 emulation.
async function listWindowsArm64ProcessNames() {
  const vendorDir = path
    .join(path.dirname(require.resolve("ps-list")), "vendor")
    .replace(/app\.asar([/\\])/, "app.asar.unpacked$1");
  const { stdout } = await promisify(execFile)(path.join(vendorDir, "fastlist-0.3.0-x64.exe"), {
    maxBuffer: 100_000_000,
    windowsHide: true,
    encoding: "utf8",
  });
  return stdout
    .trim()
    .split(/\r?\n/)
    .map((line) => line.split("\t")[2] || "");
}

async function listProcessNames() {
  if (process.platform === "win32" && process.arch === "arm64") {
    return listWindowsArm64ProcessNames();
  }
  const psList = (await import("ps-list")).default;
  const procs = await psList();
  return procs.map((p) => p.name || "");
}

class ProcessListCache {
  constructor() {
    this._cache = null;
    this._cacheTime = 0;
    this._pending = null;
  }

  async getProcessList() {
    const now = Date.now();
    if (this._cache && now - this._cacheTime < CACHE_TTL_MS) {
      return this._cache;
    }

    if (this._pending) return this._pending;

    this._pending = this._fetch(now);
    try {
      return await this._pending;
    } finally {
      this._pending = null;
    }
  }

  async _fetch(now) {
    try {
      const names = (await listProcessNames()).map((name) => name.toLowerCase());
      this._cache = names;
      this._cacheTime = now;
      debugLogger.debug("Process list refreshed", { count: names.length }, "meeting");
      return names;
    } catch (err) {
      debugLogger.warn("Failed to fetch process list", { error: err.message }, "meeting");
      return [];
    }
  }

  invalidate() {
    this._cache = null;
    this._cacheTime = 0;
    this._pending = null;
  }
}

module.exports = new ProcessListCache();
