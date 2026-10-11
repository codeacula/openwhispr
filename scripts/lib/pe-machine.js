const fs = require("fs");

// PE IMAGE_FILE_HEADER.Machine values, by Node arch name
const PE_MACHINE = Object.freeze({ x64: 0x8664, arm64: 0xaa64, ia32: 0x14c });

// Machine field of a PE image, or null for anything that is not one
function readPeMachine(filePath) {
  const fd = fs.openSync(filePath, "r");
  try {
    const dos = Buffer.alloc(64);
    if (fs.readSync(fd, dos, 0, 64, 0) < 64 || dos.readUInt16LE(0) !== 0x5a4d) return null;
    const header = Buffer.alloc(6);
    if (fs.readSync(fd, header, 0, 6, dos.readUInt32LE(0x3c)) < 6) return null;
    return header.readUInt32LE(0) === 0x00004550 ? header.readUInt16LE(4) : null;
  } finally {
    fs.closeSync(fd);
  }
}

// Whether a Windows binary was built for arch. The helper build scripts
// compile with whichever compiler they find first (an x86_64 MinGW gcc on a
// Windows on Arm machine, say), and their freshness check compares only file
// times, so without this a wrong-arch binary would be kept for good.
function isBuiltForArch(filePath, arch) {
  const expected = PE_MACHINE[arch];
  if (!expected) return true;
  try {
    return readPeMachine(filePath) === expected;
  } catch {
    return false;
  }
}

// For a helper build that produced a binary for another arch: set it aside
// while the other compilers and the download get their turn, and put it back
// only when none of them yields one for arch. It still runs emulated in
// development; afterPack fails an arm64 package that carries it.
function createWrongArchFallback(outputBinary, arch, log) {
  const asidePath = `${outputBinary}.other-arch`;
  fs.rmSync(asidePath, { force: true });
  return {
    setAside(what) {
      fs.rmSync(asidePath, { force: true });
      fs.renameSync(outputBinary, asidePath);
      log(`${what} is not built for ${arch}; trying the others`);
    },
    discard() {
      fs.rmSync(asidePath, { force: true });
    },
    restore() {
      if (!fs.existsSync(asidePath)) return false;
      fs.renameSync(asidePath, outputBinary);
      log(
        `Keeping a binary built for another arch than ${arch}: it runs emulated in development, ` +
          `but packaging for ${arch} needs a native build (compile from a Visual Studio ${arch} developer shell)`
      );
      return true;
    },
  };
}

module.exports = { PE_MACHINE, readPeMachine, isBuiltForArch, createWrongArchFallback };
