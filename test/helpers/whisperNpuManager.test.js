const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const WhisperNpuManager = require("../../src/helpers/whisperNpuManager");
const { resolveHtpArch, RUNTIME_DIR_NAME } = WhisperNpuManager;
const modelRegistryData = require("../../src/models/modelRegistryData.json");

const X_ELITE = "Snapdragon(R) X 12-core X1E80100 @ 3.40 GHz";

function tempDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "whisper-npu-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

// A Windows on Arm install: the bundled runtime under resources/bin
function armManager(t, overrides = {}) {
  const root = tempDir(t);
  const resourcesPath = path.join(root, "resources");
  const runtimeDir = path.join(resourcesPath, "bin", RUNTIME_DIR_NAME);
  fs.mkdirSync(runtimeDir, { recursive: true });
  fs.writeFileSync(path.join(runtimeDir, "whisper-npu-server.exe"), "");
  const modelsDir = path.join(root, "whisper-npu-models");
  const manager = new WhisperNpuManager({
    platform: "win32",
    arch: "arm64",
    cpuModel: X_ELITE,
    resourcesPath,
    devRoot: path.join(root, "no-dev-checkout"),
    modelsDir,
    ...overrides,
  });
  return { manager, modelsDir };
}

function installModel(modelsDir, modelName, { url, files } = {}) {
  const dir = path.join(modelsDir, modelName);
  fs.mkdirSync(dir, { recursive: true });
  for (const file of files ?? [
    "encoder.onnx",
    "encoder.bin",
    "decoder.onnx",
    "decoder.bin",
    "tokenizer.json",
  ]) {
    fs.writeFileSync(path.join(dir, file), "");
  }
  const assetUrl = url ?? modelRegistryData.whisperNpuModels[modelName].assets.v73.url;
  fs.writeFileSync(path.join(dir, ".installed.json"), JSON.stringify({ url: assetUrl }));
}

test("Snapdragon X CPUs map to their Hexagon NPU generation", () => {
  assert.equal(resolveHtpArch(X_ELITE), "v73");
  assert.equal(
    resolveHtpArch("Snapdragon(R) X Elite - X1E78100 - Qualcomm(R) Oryon(TM) CPU"),
    "v73"
  );
  assert.equal(
    resolveHtpArch("Snapdragon(R) X Plus - X1P64100 - Qualcomm(R) Oryon(TM) CPU"),
    "v73"
  );
  assert.equal(resolveHtpArch("Snapdragon(R) X2 Elite Extreme - X2E-96-100"), "v81");
  assert.equal(resolveHtpArch("Snapdragon(R) X2 Elite - X2E-88-100"), "v81");
});

test("other CPUs have no NPU target", () => {
  assert.equal(resolveHtpArch("Intel(R) Core(TM) Ultra 7 155H"), null);
  assert.equal(resolveHtpArch("Apple M3 Pro"), null);
  assert.equal(resolveHtpArch("Snapdragon (TM) 8cx Gen 3 @ 3.0 GHz"), null);
  assert.equal(resolveHtpArch(undefined), null);
});

test("only Windows on Arm with the bundled runtime and a Snapdragon X is supported", (t) => {
  assert.equal(armManager(t).manager.isSupported(), true);
  assert.equal(armManager(t, { arch: "x64" }).manager.isSupported(), false);
  assert.equal(armManager(t, { platform: "linux" }).manager.isSupported(), false);
  assert.equal(armManager(t, { cpuModel: "Intel(R) Core(TM) i7" }).manager.isSupported(), false);
  assert.equal(armManager(t, { resourcesPath: tempDir(t) }).manager.isSupported(), false);
});

test("an unsupported machine reports no NPU without running the probe", async (t) => {
  const { manager } = armManager(t, { arch: "x64" });
  assert.deepEqual(await manager.probe(), { npu: false });
  assert.equal(manager.probePromise, null);
});

test("every NPU build pairs a v73 and a v81 asset with a pinned tokenizer", () => {
  for (const [name, config] of Object.entries(modelRegistryData.whisperNpuModels)) {
    assert.ok(modelRegistryData.whisperModels[name], `${name} is a known Whisper model`);
    assert.match(config.tokenizer.sha256, /^[0-9a-f]{64}$/, name);
    for (const arch of ["v73", "v81"]) {
      assert.match(config.assets[arch].url, /^https:\/\//, `${name} ${arch}`);
      assert.ok(config.assets[arch].sizeBytes > 0, `${name} ${arch}`);
    }
  }
});

test("models without an NPU build are not offered", (t) => {
  const { manager } = armManager(t);
  assert.equal(manager.hasModel("small"), true);
  assert.equal(manager.hasModel("turbo"), true);
  assert.equal(manager.hasModel("large"), false);
  assert.throws(() => manager.getModelDir("../../escape"));
});

test("a model counts as installed only with every file and this NPU's asset", (t) => {
  const { manager, modelsDir } = armManager(t);
  assert.equal(manager.isModelDownloaded("small"), false);

  installModel(modelsDir, "small");
  assert.equal(manager.isModelDownloaded("small"), true);

  // Interrupted install: the decoder weights never arrived
  installModel(modelsDir, "base", { files: ["encoder.onnx", "encoder.bin", "decoder.onnx"] });
  assert.equal(manager.isModelDownloaded("base"), false);

  // Built for the X2's v81 NPU, copied onto an X Elite
  installModel(modelsDir, "tiny", {
    url: modelRegistryData.whisperNpuModels.tiny.assets.v81.url,
  });
  assert.equal(manager.isModelDownloaded("tiny"), false);
});

test("status describes the model on this machine's NPU", async (t) => {
  const { manager, modelsDir } = armManager(t);
  manager.probe = async () => ({ npu: true });
  installModel(modelsDir, "small");

  const small = await manager.getStatus("small");
  assert.equal(small.supported, true);
  assert.equal(small.npuAvailable, true);
  assert.equal(small.htpArch, "v73");
  assert.equal(small.modelSupported, true);
  assert.equal(small.downloaded, true);
  assert.equal(
    small.downloadSizeBytes,
    modelRegistryData.whisperNpuModels.small.assets.v73.sizeBytes
  );

  const large = await manager.getStatus("large");
  assert.equal(large.modelSupported, false);
  assert.equal(large.downloaded, false);
});

test("deleting a model removes its folder", async (t) => {
  const { manager, modelsDir } = armManager(t);
  installModel(modelsDir, "small");
  assert.deepEqual(await manager.deleteModel("small"), {
    success: true,
    deleted: true,
    model: "small",
  });
  assert.equal(fs.existsSync(path.join(modelsDir, "small")), false);
});
