#!/usr/bin/env node
/**
 * Ensures the Windows fast-paste binary is available.
 *
 * Strategy:
 * 1. If binary exists and is up-to-date, do nothing
 * 2. Try to download prebuilt binary from GitHub releases
 * 3. Fall back to local compilation if download fails
 *
 * This allows developers without a C compiler to still build the app.
 */

const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");
const { isBuiltForArch, createWrongArchFallback } = require("./lib/pe-machine");

const isWindows = process.platform === "win32";
if (!isWindows) {
  // Only needed on Windows
  process.exit(0);
}

const projectRoot = path.resolve(__dirname, "..");
const cSource = path.join(projectRoot, "resources", "windows-fast-paste.c");
const outputDir = path.join(projectRoot, "resources", "bin");
// TARGET_ARCH is set per leg by the Windows CI matrix
const targetArch = process.env.TARGET_ARCH || process.arch;
const outputBinary = path.join(outputDir, "windows-fast-paste.exe");

function log(message) {
  console.log(`[windows-fast-paste] ${message}`);
}

const wrongArch = createWrongArchFallback(outputBinary, targetArch, log);

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function isBinaryUpToDate() {
  if (!fs.existsSync(outputBinary)) {
    return false;
  }

  if (!fs.existsSync(cSource)) {
    return true;
  }

  try {
    const binaryStat = fs.statSync(outputBinary);
    const sourceStat = fs.statSync(cSource);
    return binaryStat.mtimeMs >= sourceStat.mtimeMs && isUsableBinary(outputBinary);
  } catch {
    return false;
  }
}

// Built for the target arch (the capability probe below also passes for an
// x64 binary running emulated on Windows on Arm) and new enough
function isUsableBinary(binaryPath) {
  return isBuiltForArch(binaryPath, targetArch) && supportsSelectionCopy(binaryPath);
}

function supportsSelectionCopy(binaryPath) {
  // The capability probe executes the binary, which is only possible on
  // Windows itself; when cross-packaging, trust the downloaded artifact.
  if (process.platform !== "win32") {
    return true;
  }
  try {
    const result = spawnSync(binaryPath, ["--capabilities"], {
      stdio: "pipe",
      windowsHide: true,
    });
    return result.status === 0 && result.stdout.toString().includes("selection-copy-v1");
  } catch {
    return false;
  }
}

// A stale binary must never linger at the output path: packaging would ship
// it, and its missing selection-copy support silently disables selection
// editing at runtime.
function discardStaleBinary(reason) {
  try {
    fs.rmSync(outputBinary, { force: true });
    log(reason);
  } catch {}
}

async function tryDownload() {
  log("Attempting to download prebuilt binary...");

  const downloadScript = path.join(__dirname, "download-windows-fast-paste.js");
  if (!fs.existsSync(downloadScript)) {
    log("Download script not found, skipping download");
    return false;
  }

  // A download that finds no asset for this arch keeps whatever is on disk, so
  // a stale binary (another arch, an older build) would pass as downloaded and
  // stop the compile below. It is out of date anyway, or main() returned early.
  if (fs.existsSync(outputBinary)) {
    discardStaleBinary("Removed outdated fast-paste binary before downloading");
  }

  const result = spawnSync(process.execPath, [downloadScript, "--force"], {
    stdio: "inherit",
    cwd: projectRoot,
  });

  if (result.status === 0 && fs.existsSync(outputBinary)) {
    if (isUsableBinary(outputBinary)) {
      log("Successfully downloaded prebuilt binary");
      return true;
    }
    if (supportsSelectionCopy(outputBinary)) {
      wrongArch.setAside("Downloaded binary");
      return false;
    }
    discardStaleBinary(
      "Discarded prebuilt binary without selection-copy-v1 support; compiling locally"
    );
    return false;
  }

  log("Download failed or binary not found after download");
  return false;
}

/**
 * Quote a path for use in shell commands on Windows.
 * @param {string} p - Path to quote
 * @returns {string} - Quoted path safe for shell use
 */
function quotePath(p) {
  return `"${p.replace(/"/g, '\\"')}"`;
}

function tryCompile() {
  if (!fs.existsSync(cSource)) {
    log("C source not found, cannot compile locally");
    return false;
  }

  log("Attempting local compilation...");

  const compilers = [
    {
      name: "MSVC",
      check: { command: "cl", args: [] },
      useShell: true,
      getCommand: () =>
        `cl /O2 /nologo ${quotePath(cSource)} /Fe:${quotePath(outputBinary)} user32.lib`,
    },
    {
      name: "MinGW-w64",
      check: { command: "gcc", args: ["--version"] },
      useShell: false,
      command: "gcc",
      args: ["-O2", cSource, "-o", outputBinary, "-luser32"],
    },
    {
      name: "Clang",
      check: { command: "clang", args: ["--version"] },
      useShell: false,
      command: "clang",
      args: ["-O2", cSource, "-o", outputBinary, "-luser32"],
    },
  ];

  for (const compiler of compilers) {
    log(`Trying ${compiler.name}...`);

    const checkResult = spawnSync(compiler.check.command, compiler.check.args, {
      stdio: "pipe",
      shell: true,
    });

    if (checkResult.status !== 0 && checkResult.error) {
      log(`${compiler.name} not found, trying next...`);
      continue;
    }

    let result;
    if (compiler.useShell) {
      const cmd = compiler.getCommand();
      log(`Compiling with: ${cmd}`);
      result = spawnSync(cmd, [], {
        stdio: "inherit",
        cwd: projectRoot,
        shell: true,
      });
    } else {
      log(`Compiling with: ${compiler.command} ${compiler.args.join(" ")}`);
      result = spawnSync(compiler.command, compiler.args, {
        stdio: "inherit",
        cwd: projectRoot,
        shell: false,
      });
    }

    if (result.status === 0 && fs.existsSync(outputBinary)) {
      if (isUsableBinary(outputBinary)) {
        log(`Successfully built with ${compiler.name}`);
        return true;
      }
      if (supportsSelectionCopy(outputBinary)) {
        wrongArch.setAside(`Binary built with ${compiler.name}`);
        continue;
      }
      discardStaleBinary(`Binary built with ${compiler.name} lacks selection-copy-v1; discarded`);
    }

    log(`${compiler.name} compilation failed, trying next...`);
  }

  return false;
}

async function main() {
  ensureDir(outputDir);

  if (isBinaryUpToDate()) {
    log("Binary is up to date, skipping build");
    return;
  }

  // A working binary for another arch is kept only if nothing below yields a
  // binary for targetArch (tryDownload would otherwise delete it)
  if (
    fs.existsSync(outputBinary) &&
    !isBuiltForArch(outputBinary, targetArch) &&
    supportsSelectionCopy(outputBinary)
  ) {
    wrongArch.setAside("Existing binary");
  }

  const obtained = (await tryDownload()) || tryCompile();
  if (obtained) {
    wrongArch.discard();
    return;
  }

  if (fs.existsSync(outputBinary) && !isUsableBinary(outputBinary)) {
    discardStaleBinary("Removed stale fast-paste binary so packaging cannot ship it");
  }
  if (wrongArch.restore()) {
    return;
  }

  console.warn("[windows-fast-paste] Could not obtain Windows fast-paste binary.");
  console.warn(
    "[windows-fast-paste] Windows paste will use nircmd/PowerShell fallback; selection editing will type at the cursor."
  );
}

main().catch((error) => {
  console.error("[windows-fast-paste] Unexpected error:", error);
  // Don't fail the build
});
