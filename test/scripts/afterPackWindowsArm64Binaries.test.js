const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { Arch } = require("app-builder-lib");

const {
  verifyWindowsArm64Binaries,
  findForeignArchBinaries,
  readPeMachine,
} = require("../../scripts/afterPack");

const ARM64 = 0xaa64;
const X64 = 0x8664;
const X86 = 0x14c;

// The smallest file readPeMachine accepts: an MZ header pointing at a PE header
function writePe(file, machine) {
  const image = Buffer.alloc(0x86);
  image.writeUInt16LE(0x5a4d, 0); // "MZ"
  image.writeUInt32LE(0x80, 0x3c); // e_lfanew
  image.writeUInt32LE(0x00004550, 0x80); // "PE\0\0"
  image.writeUInt16LE(machine, 0x84);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, image);
}

function makeAppOutDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "afterpack-arm64-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  writePe(path.join(dir, "OpenWhispr.exe"), ARM64);
  writePe(path.join(dir, "resources", "bin", "whisper-server-win32-arm64.exe"), ARM64);
  writePe(path.join(dir, "resources", "bin", "vcruntime140.dll"), ARM64);
  // Deliberately emulated: nothing upstream ships them for arm64
  writePe(path.join(dir, "resources", "bin", "qdrant-win32-arm64.exe"), X64);
  writePe(path.join(dir, "resources", "bin", "nircmd.exe"), X64);
  writePe(path.join(dir, "resources", "elevate.exe"), X86);
  const vendor = path.join(dir, "resources", "app.asar.unpacked", "node_modules");
  writePe(path.join(vendor, "ffmpeg-static", "ffmpeg.exe"), X64);
  writePe(path.join(vendor, "ps-list", "vendor", "fastlist-0.3.0-x64.exe"), X64);
  writePe(path.join(vendor, "ps-list", "vendor", "fastlist-0.3.0-x86.exe"), X86);
  writePe(path.join(vendor, "better-sqlite3", "build", "Release", "better_sqlite3.node"), ARM64);
  // Not PE images, so never checked: the NPU pack's Hexagon libraries and licenses
  fs.mkdirSync(path.join(dir, "resources", "bin", "whisper-npu-win32-arm64"), { recursive: true });
  fs.writeFileSync(
    path.join(dir, "resources", "bin", "whisper-npu-win32-arm64", "libQnnHtpV73Skel.so"),
    "\x7fELF"
  );
  return dir;
}

function context(appOutDir, arch = "arm64", platform = "win32") {
  return { appOutDir, electronPlatformName: platform, arch: Arch[arch] };
}

test("reads the machine of PE images and ignores everything else", (t) => {
  const dir = makeAppOutDir(t);
  assert.equal(readPeMachine(path.join(dir, "resources", "bin", "nircmd.exe")), X64);
  assert.equal(readPeMachine(path.join(dir, "OpenWhispr.exe")), ARM64);
  const npu = path.join(dir, "resources", "bin", "whisper-npu-win32-arm64", "libQnnHtpV73Skel.so");
  assert.equal(readPeMachine(npu), null);
});

test("an arm64 build of ARM64 binaries plus the known emulated ones passes", (t) => {
  assert.doesNotThrow(() => verifyWindowsArm64Binaries(context(makeAppOutDir(t))));
});

test("an x64 helper or DLL in an arm64 build fails it, naming the file", (t) => {
  const dir = makeAppOutDir(t);
  // A helper restored from the x64 cache, and VS's arm64-folder vcruntime140_1
  // (ARM64EC, so its header reads x64)
  writePe(path.join(dir, "resources", "bin", "windows-key-listener.exe"), X64);
  writePe(path.join(dir, "resources", "bin", "vcruntime140_1.dll"), X64);
  assert.throws(
    () => verifyWindowsArm64Binaries(context(dir)),
    (error) =>
      /windows-key-listener\.exe \(0x8664\)/.test(error.message) &&
      /vcruntime140_1\.dll \(0x8664\)/.test(error.message)
  );
});

test("an x64 native module in an arm64 build fails it", (t) => {
  const dir = makeAppOutDir(t);
  const keyring = path.join(
    dir,
    "resources",
    "app.asar.unpacked",
    "node_modules",
    "@napi-rs",
    "keyring-win32-x64-msvc",
    "keyring.win32-x64-msvc.node"
  );
  writePe(keyring, X64);
  assert.deepEqual(
    findForeignArchBinaries(
      dir,
      ARM64,
      new Set([
        "nircmd.exe",
        "qdrant-win32-arm64.exe",
        "ffmpeg.exe",
        "fastlist-0.3.0-x64.exe",
        "fastlist-0.3.0-x86.exe",
        "elevate.exe",
      ])
    ).map(({ file }) => path.basename(file)),
    ["keyring.win32-x64-msvc.node"]
  );
});

test("x64 Windows, macOS and Linux builds are not checked", (t) => {
  const dir = makeAppOutDir(t);
  writePe(path.join(dir, "resources", "bin", "windows-key-listener.exe"), X64);
  assert.doesNotThrow(() => verifyWindowsArm64Binaries(context(dir, "x64")));
  assert.doesNotThrow(() => verifyWindowsArm64Binaries(context(dir, "arm64", "darwin")));
  assert.doesNotThrow(() => verifyWindowsArm64Binaries(context(dir, "arm64", "linux")));
});
