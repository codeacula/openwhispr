#!/usr/bin/env node
/**
 * Builds whisper-npu-server (native/whisper-npu-server): Whisper on the
 * Snapdragon Hexagon NPU through ONNX Runtime's QNN execution provider.
 *
 * Stages the exe with everything it loads into
 * resources/bin/whisper-npu-win32-arm64/: ONNX Runtime, the QNN provider and
 * HTP libraries for Snapdragon X (v73) and X2 (v81), the MSVC runtime that
 * onnxruntime.dll links, and their licenses. QnnHtpPrepare.dll (170 MB) is
 * left out: it compiles models on the device, and the server only loads
 * models precompiled by Qualcomm AI Hub.
 *
 * Windows on Arm only; exits quietly anywhere else.
 *
 *   node scripts/build-whisper-npu-server.js [--force] [--zip]
 *
 * --zip also writes whisper-npu-server-win32-arm64.zip for a release.
 */

const crypto = require("crypto");
const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const { downloadFile, extractZip } = require("./lib/download-utils");

const archIndex = process.argv.indexOf("--arch");
const targetArch =
  (archIndex !== -1 && process.argv[archIndex + 1]) || process.env.TARGET_ARCH || process.arch;

if (process.platform !== "win32" || targetArch !== "arm64") {
  console.log("[whisper-npu-server] Skipping: Windows on Arm only");
  process.exit(0);
}

const projectRoot = path.resolve(__dirname, "..");
const helperRoot = path.join(projectRoot, "native", "whisper-npu-server");
const thirdPartyRoot = path.join(helperRoot, "third_party");
const buildDir = path.join(helperRoot, "build", "win32-arm64");
const packName = "whisper-npu-win32-arm64";
const packDir = path.join(projectRoot, "resources", "bin", packName);
const hashFile = path.join(packDir, ".build-hash");

// ONNX Runtime and the QNN provider must match the versions the AI Hub
// models were compiled for (ONNX Runtime 1.30, QAIRT 2.50 in AI Hub v0.64).
const PINS = {
  httplib: {
    url: "https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.20.0/httplib.h",
    sha256: "2831f48258d8e20ea88ea82c63f15014ced0a584cbdb1cd3bea1a60b6fb971e5",
    file: "include/httplib.h",
  },
  json: {
    url: "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp",
    sha256: "9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6",
    file: "include/json.hpp",
  },
  onnxruntime: {
    url: "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/1.30.0/microsoft.ml.onnxruntime.1.30.0.nupkg",
    sha256: "30df119bd9164eaa187666d93147448fdd4a9a1a3caf83f1505daf7af2d1c5d2",
    dir: "onnxruntime-1.30.0",
  },
  qnn: {
    url: "https://api.nuget.org/v3-flatcontainer/qualcomm.ml.onnxruntime.qnn/2.6.0/qualcomm.ml.onnxruntime.qnn.2.6.0.nupkg",
    sha256: "c2fe66eeaf92a0cb89faef1d4d05c23b939896aeb9dcfa2261d8fe70b54b8103",
    dir: "qnn-2.6.0",
  },
};

const ORT_NATIVE = "runtimes/win-arm64/native";
// [source relative to the package, name in the pack]
const RUNTIME_FILES = {
  onnxruntime: [
    [`${ORT_NATIVE}/onnxruntime.dll`, "onnxruntime.dll"],
    ["LICENSE", "ONNXRUNTIME_LICENSE.txt"],
    ["ThirdPartyNotices.txt", "ONNXRUNTIME_ThirdPartyNotices.txt"],
  ],
  qnn: [
    [`${ORT_NATIVE}/onnxruntime_providers_qnn.dll`, "onnxruntime_providers_qnn.dll"],
    [`${ORT_NATIVE}/QnnHtp.dll`, "QnnHtp.dll"],
    [`${ORT_NATIVE}/QnnSystem.dll`, "QnnSystem.dll"],
    [`${ORT_NATIVE}/QnnHtpV73Stub.dll`, "QnnHtpV73Stub.dll"],
    [`${ORT_NATIVE}/libQnnHtpV73Skel.so`, "libQnnHtpV73Skel.so"],
    [`${ORT_NATIVE}/libqnnhtpv73.cat`, "libqnnhtpv73.cat"],
    [`${ORT_NATIVE}/QnnHtpV81Stub.dll`, "QnnHtpV81Stub.dll"],
    [`${ORT_NATIVE}/libQnnHtpV81Skel.so`, "libQnnHtpV81Skel.so"],
    [`${ORT_NATIVE}/libqnnhtpv81.cat`, "libqnnhtpv81.cat"],
    ["LICENSE", "QNN_EP_LICENSE.txt"],
    ["Qualcomm_LICENSE.pdf", "Qualcomm_LICENSE.pdf"],
  ],
};
const CRT_LIBRARIES = ["msvcp140.dll", "msvcp140_1.dll", "vcruntime140.dll", "vcruntime140_1.dll"];

function log(message) {
  console.log(`[whisper-npu-server] ${message}`);
}

function sha256(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function run(command, args) {
  log(`Running ${command} ${args.join(" ")}`);
  return spawnSync(command, args, { stdio: "inherit" });
}

async function fetchPinned(pin, destination) {
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  await downloadFile(pin.url, destination);
  const actual = sha256(destination);
  if (actual !== pin.sha256) {
    fs.rmSync(destination, { force: true });
    throw new Error(`${pin.url} sha256 mismatch: ${actual}`);
  }
}

async function ensureThirdParty() {
  for (const pin of [PINS.httplib, PINS.json]) {
    const target = path.join(thirdPartyRoot, pin.file);
    if (!fs.existsSync(target) || sha256(target) !== pin.sha256) {
      log(`Downloading ${pin.url}`);
      await fetchPinned(pin, target);
    }
  }
  for (const pin of [PINS.onnxruntime, PINS.qnn]) {
    const target = path.join(thirdPartyRoot, pin.dir);
    const marker = path.join(target, ".sha256");
    if (fs.existsSync(marker) && fs.readFileSync(marker, "utf8") === pin.sha256) continue;
    log(`Downloading ${pin.url}`);
    const archive = `${target}.zip`;
    try {
      await fetchPinned(pin, archive);
      fs.rmSync(target, { recursive: true, force: true });
      await extractZip(archive, target);
      fs.writeFileSync(marker, pin.sha256);
    } finally {
      fs.rmSync(archive, { force: true });
    }
  }
}

// MSVC's arm64 CRT redist folder: VCToolsRedistDir inside a developer shell
// (ilammy/msvc-dev-cmd in CI), otherwise the newest Visual Studio install.
function findCrtDir() {
  const roots = [];
  if (process.env.VCToolsRedistDir) roots.push(process.env.VCToolsRedistDir);
  const vswhere = path.join(
    process.env["ProgramFiles(x86)"] || "C:\\Program Files (x86)",
    "Microsoft Visual Studio",
    "Installer",
    "vswhere.exe"
  );
  if (fs.existsSync(vswhere)) {
    const result = spawnSync(
      vswhere,
      ["-latest", "-products", "*", "-property", "installationPath"],
      { encoding: "utf8" }
    );
    const install = (result.stdout || "").trim();
    const redist = path.join(install, "VC", "Redist", "MSVC");
    if (install && fs.existsSync(redist)) {
      for (const version of fs.readdirSync(redist).sort().reverse()) {
        roots.push(path.join(redist, version));
      }
    }
  }
  for (const root of roots) {
    const arm64 = path.join(root, "arm64");
    if (!fs.existsSync(arm64)) continue;
    const crt = fs.readdirSync(arm64).find((name) => /^Microsoft\.VC\d+\.CRT$/.test(name));
    if (crt) return path.join(arm64, crt);
  }
  throw new Error("Cannot find the arm64 MSVC runtime redist; run from a VS developer shell");
}

function sourceHash() {
  const hash = crypto.createHash("sha256");
  hash.update(JSON.stringify(PINS));
  hash.update(JSON.stringify(RUNTIME_FILES));
  const files = [
    path.join(helperRoot, "CMakeLists.txt"),
    ...fs
      .readdirSync(path.join(helperRoot, "src"))
      .sort()
      .map((name) => path.join(helperRoot, "src", name)),
  ];
  for (const file of files) {
    hash.update(path.relative(helperRoot, file));
    hash.update(fs.readFileSync(file));
  }
  return hash.digest("hex");
}

function stagePack(builtExe) {
  fs.rmSync(packDir, { recursive: true, force: true });
  fs.mkdirSync(packDir, { recursive: true });
  fs.copyFileSync(builtExe, path.join(packDir, "whisper-npu-server.exe"));
  for (const [key, files] of Object.entries(RUNTIME_FILES)) {
    const packageRoot = path.join(thirdPartyRoot, PINS[key].dir);
    for (const [source, name] of files) {
      fs.copyFileSync(path.join(packageRoot, source), path.join(packDir, name));
    }
  }
  const crtDir = findCrtDir();
  for (const name of CRT_LIBRARIES) {
    fs.copyFileSync(path.join(crtDir, name), path.join(packDir, name));
  }
}

function zipPack() {
  const zipPath = path.join(buildDir, "whisper-npu-server-win32-arm64.zip");
  fs.rmSync(zipPath, { force: true });
  // Windows' bundled bsdtar writes zip archives with -a
  const result = run("tar", ["-a", "-c", "-f", zipPath, "-C", packDir, "."]);
  if (result.status !== 0) throw new Error("Failed to zip the pack");
  log(`Wrote ${path.relative(projectRoot, zipPath)}`);
}

async function main() {
  const force = process.argv.includes("--force");
  const currentHash = sourceHash();
  const builtExe = path.join(buildDir, "Release", "whisper-npu-server.exe");
  if (
    !force &&
    fs.existsSync(hashFile) &&
    fs.readFileSync(hashFile, "utf8") === currentHash &&
    fs.existsSync(path.join(packDir, "whisper-npu-server.exe"))
  ) {
    log("Pack is up to date, skipping build");
    if (process.argv.includes("--zip")) zipPack();
    return;
  }

  await ensureThirdParty();

  const configure = run("cmake", [
    "-S",
    helperRoot,
    "-B",
    buildDir,
    "-A",
    "ARM64",
    `-DORT_INCLUDE_DIR=${path.join(thirdPartyRoot, PINS.onnxruntime.dir, "build", "native", "include")}`,
    `-DTHIRD_PARTY_INCLUDE_DIR=${path.join(thirdPartyRoot, "include")}`,
  ]);
  if (configure.status !== 0) throw new Error("CMake configure failed");
  const build = run("cmake", ["--build", buildDir, "--config", "Release"]);
  if (build.status !== 0 || !fs.existsSync(builtExe)) throw new Error("Build failed");

  stagePack(builtExe);
  fs.writeFileSync(hashFile, currentHash);
  log(`Staged ${path.relative(projectRoot, packDir)}`);
  if (process.argv.includes("--zip")) zipPack();
}

main().catch((error) => {
  console.error(`[whisper-npu-server] ${error.message}`);
  process.exitCode = 1;
});
