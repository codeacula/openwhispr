const crypto = require("crypto");
const { execFile } = require("child_process");
const fs = require("fs");
const fsPromises = require("fs").promises;
const os = require("os");
const path = require("path");
const debugLogger = require("./debugLogger");
const {
  downloadFile,
  createDownloadSignal,
  createDownloadInProgressError,
  validateFileSize,
  checkDiskSpace,
  extractArchive,
  findFile,
} = require("./downloadUtils");
const { getModelsDirForService } = require("./modelDirUtils");
const modelRegistryData = require("../models/modelRegistryData.json");

// Whisper on the Snapdragon Hexagon NPU. The runtime (whisper-npu-server with
// ONNX Runtime and Qualcomm's QNN libraries) ships inside the Windows on Arm
// installer, built by scripts/build-whisper-npu-server.js; Qualcomm's license
// allows redistributing it only as part of the app. The models are Qualcomm
// AI Hub "precompiled_qnn_onnx" exports of the same Whisper checkpoints,
// downloaded per model from Qualcomm and installed beside a pinned tokenizer.

const RUNTIME_DIR_NAME = "whisper-npu-win32-arm64";
const SERVER_BINARY = "whisper-npu-server.exe";
const MODEL_FILES = ["encoder.onnx", "encoder.bin", "decoder.onnx", "decoder.bin"];
const TOKENIZER_FILE = "tokenizer.json";
const INSTALL_MARKER = ".installed.json";
const PROBE_TIMEOUT_MS = 20000;
// The zip and its extracted copy coexist until the install finishes
const DISK_SPACE_FACTOR = 2.2;

// Snapdragon X Elite/Plus (X1E, X1P) carry a v73 Hexagon NPU, Snapdragon X2
// a v81. AI Hub compiles each model per NPU generation.
function resolveHtpArch(cpuModel) {
  const name = String(cpuModel || "");
  if (/\bX2[EP]?-?\d|\bX2\b/i.test(name)) return "v81";
  if (/\bX1[EP]-?\d|Snapdragon\(R\) X\b/i.test(name)) return "v73";
  return null;
}

function getNpuModelConfig(modelName) {
  return modelRegistryData.whisperNpuModels?.[modelName] || null;
}

function sha256File(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

class WhisperNpuManager {
  constructor({
    platform = process.platform,
    arch = process.arch,
    cpuModel = os.cpus()[0]?.model,
    resourcesPath = process.resourcesPath,
    devRoot = path.join(__dirname, "..", ".."),
    modelsDir = null,
  } = {}) {
    this.platform = platform;
    this.arch = arch;
    this.htpArch = resolveHtpArch(cpuModel);
    this.resourcesPath = resourcesPath;
    this.devRoot = devRoot;
    this.modelsDirOverride = modelsDir;
    this.probePromise = null;
    this.currentDownload = null;
  }

  getRuntimeDir() {
    const candidates = [];
    if (this.resourcesPath) candidates.push(path.join(this.resourcesPath, "bin", RUNTIME_DIR_NAME));
    candidates.push(path.join(this.devRoot, "resources", "bin", RUNTIME_DIR_NAME));
    return candidates.find((dir) => fs.existsSync(path.join(dir, SERVER_BINARY))) || null;
  }

  getServerBinaryPath() {
    const dir = this.getRuntimeDir();
    return dir ? path.join(dir, SERVER_BINARY) : null;
  }

  // The bundled runtime exists and the CPU is a Snapdragon AI Hub targets.
  // Whether the NPU itself is usable is probe()'s job.
  isSupported() {
    return (
      this.platform === "win32" &&
      this.arch === "arm64" &&
      this.htpArch !== null &&
      this.getServerBinaryPath() !== null
    );
  }

  // Asks the runtime whether ONNX Runtime's QNN provider sees an NPU (it does
  // not without Qualcomm's NPU driver). Cached: hardware does not change
  // under a running app.
  probe() {
    if (!this.isSupported()) return Promise.resolve({ npu: false });
    if (this.probePromise) return this.probePromise;
    const binary = this.getServerBinaryPath();
    this.probePromise = new Promise((resolve) => {
      execFile(
        binary,
        ["--probe"],
        { cwd: path.dirname(binary), timeout: PROBE_TIMEOUT_MS, windowsHide: true },
        (error, stdout) => {
          let npu = false;
          try {
            npu = JSON.parse(String(stdout).trim().split("\n").pop()).npu === true;
          } catch {
            // No JSON: the runtime failed to load
          }
          if (!npu) {
            debugLogger.info("Whisper NPU probe found no usable NPU", {
              exitCode: error?.code ?? 0,
              stdout: String(stdout).slice(0, 500),
            });
          }
          resolve({ npu });
        }
      );
    });
    return this.probePromise;
  }

  getModelsDir() {
    return this.modelsDirOverride || getModelsDirForService("whisper-npu");
  }

  // NPU builds exist for this model on this machine's NPU generation
  hasModel(modelName) {
    return !!(this.htpArch && getNpuModelConfig(modelName)?.assets?.[this.htpArch]);
  }

  getModelDir(modelName) {
    if (!getNpuModelConfig(modelName)) throw new Error(`No NPU build of Whisper "${modelName}"`);
    return path.join(this.getModelsDir(), modelName);
  }

  getAsset(modelName) {
    return this.hasModel(modelName) ? getNpuModelConfig(modelName).assets[this.htpArch] : null;
  }

  isModelDownloaded(modelName) {
    const asset = this.getAsset(modelName);
    if (!asset) return false;
    const dir = this.getModelDir(modelName);
    try {
      const marker = JSON.parse(fs.readFileSync(path.join(dir, INSTALL_MARKER), "utf8"));
      return (
        marker.url === asset.url &&
        [...MODEL_FILES, TOKENIZER_FILE].every((file) => fs.existsSync(path.join(dir, file)))
      );
    } catch {
      return false;
    }
  }

  async getStatus(modelName) {
    const supported = this.isSupported();
    const { npu } = supported ? await this.probe() : { npu: false };
    const asset = supported ? this.getAsset(modelName) : null;
    const download = this.currentDownload;
    return {
      supported,
      npuAvailable: npu,
      htpArch: this.htpArch,
      model: modelName,
      modelSupported: !!asset,
      downloaded: !!asset && this.isModelDownloaded(modelName),
      downloadSizeBytes: asset ? asset.sizeBytes : 0,
      downloading: download?.model === modelName,
      downloadProgress: download?.model === modelName ? download.percentage : 0,
    };
  }

  async downloadModel(modelName, onProgress = () => {}) {
    const asset = this.getAsset(modelName);
    if (!asset) throw new Error(`No NPU build of Whisper "${modelName}" for this device`);
    if (this.isModelDownloaded(modelName)) return { success: true, model: modelName };
    if (this.currentDownload) {
      throw createDownloadInProgressError(modelName, this.currentDownload.model);
    }

    const { signal, abort } = createDownloadSignal();
    const download = { model: modelName, abort, percentage: 0 };
    this.currentDownload = download;

    const modelsDir = this.getModelsDir();
    const zipPath = path.join(modelsDir, `${modelName}-npu.zip`);
    const stagingDir = path.join(modelsDir, `.${modelName}.staging`);
    const modelDir = this.getModelDir(modelName);
    const tokenizer = getNpuModelConfig(modelName).tokenizer;

    try {
      await fsPromises.mkdir(modelsDir, { recursive: true });
      const needed = asset.sizeBytes * DISK_SPACE_FACTOR;
      const space = await checkDiskSpace(modelsDir, needed);
      if (!space.ok) {
        throw new Error(
          `Not enough disk space for the NPU model. Need ~${Math.round(needed / 1_000_000)}MB, ` +
            `only ${Math.round(space.availableBytes / 1_000_000)}MB available.`
        );
      }

      await downloadFile(asset.url, zipPath, {
        timeout: 600000,
        signal,
        expectedSize: asset.sizeBytes,
        onProgress: (downloaded, total) => {
          download.percentage = total > 0 ? Math.round((downloaded / total) * 100) : 0;
          onProgress({
            type: "progress",
            model: modelName,
            downloaded_bytes: downloaded,
            total_bytes: total,
            percentage: download.percentage,
          });
        },
      });
      await validateFileSize(zipPath, asset.sizeBytes);
      if (signal.aborted) throw Object.assign(new Error("Download cancelled"), { isAbort: true });

      onProgress({ type: "installing", model: modelName, percentage: 100 });
      await fsPromises.rm(stagingDir, { recursive: true, force: true });
      await fsPromises.mkdir(stagingDir, { recursive: true });
      await extractArchive(zipPath, stagingDir);

      // The archive nests the model files one folder down
      const encoder = await findFile(stagingDir, "encoder.onnx");
      if (!encoder) throw new Error("NPU model archive has no encoder.onnx");
      const extracted = path.dirname(encoder);
      const missing = MODEL_FILES.filter((file) => !fs.existsSync(path.join(extracted, file)));
      if (missing.length > 0) {
        throw new Error(`NPU model archive is missing ${missing.join(", ")}`);
      }

      const tokenizerPath = path.join(extracted, TOKENIZER_FILE);
      await downloadFile(tokenizer.url, tokenizerPath, {
        signal,
        expectedSize: tokenizer.sizeBytes,
      });
      const digest = sha256File(tokenizerPath);
      if (digest !== tokenizer.sha256) throw new Error(`Tokenizer sha256 mismatch: ${digest}`);

      await fsPromises.writeFile(
        path.join(extracted, INSTALL_MARKER),
        JSON.stringify({ url: asset.url, htpArch: this.htpArch, installedAt: Date.now() })
      );
      await fsPromises.rm(modelDir, { recursive: true, force: true });
      await fsPromises.rename(extracted, modelDir);

      onProgress({ type: "complete", model: modelName, percentage: 100 });
      return { success: true, model: modelName };
    } catch (error) {
      if (error.isAbort || signal.aborted) {
        await fsPromises.rm(`${zipPath}.tmp`, { force: true }).catch(() => {});
        throw Object.assign(new Error("Download interrupted by user"), {
          code: "DOWNLOAD_CANCELLED",
        });
      }
      throw error;
    } finally {
      await fsPromises.rm(zipPath, { force: true }).catch(() => {});
      await fsPromises.rm(stagingDir, { recursive: true, force: true }).catch(() => {});
      if (this.currentDownload === download) this.currentDownload = null;
    }
  }

  cancelDownload() {
    if (!this.currentDownload) return { success: false, error: "No active download to cancel" };
    this.currentDownload.abort();
    return { success: true };
  }

  async deleteModel(modelName) {
    const dir = this.getModelDir(modelName);
    const existed = fs.existsSync(dir);
    await fsPromises.rm(dir, { recursive: true, force: true });
    return { success: true, deleted: existed, model: modelName };
  }
}

module.exports = WhisperNpuManager;
module.exports.resolveHtpArch = resolveHtpArch;
module.exports.RUNTIME_DIR_NAME = RUNTIME_DIR_NAME;
