const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("fs");
const os = require("os");
const path = require("path");

const {
  PE_MACHINE,
  readPeMachine,
  isBuiltForArch,
  createWrongArchFallback,
} = require("../../scripts/lib/pe-machine");

// The smallest file readPeMachine accepts: an MZ header pointing at a PE header
function writePe(file, machine) {
  const image = Buffer.alloc(0x86);
  image.writeUInt16LE(0x5a4d, 0); // "MZ"
  image.writeUInt32LE(0x80, 0x3c); // e_lfanew
  image.writeUInt32LE(0x00004550, 0x80); // "PE\0\0"
  image.writeUInt16LE(machine, 0x84);
  fs.writeFileSync(file, image);
}

function tempDir(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pe-machine-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

test("a binary is built for the arch its PE header names", (t) => {
  const exe = path.join(tempDir(t), "helper.exe");
  writePe(exe, PE_MACHINE.arm64);
  assert.equal(readPeMachine(exe), 0xaa64);
  assert.equal(isBuiltForArch(exe, "arm64"), true);
  assert.equal(isBuiltForArch(exe, "x64"), false);
});

test("a missing or non-PE file is built for no arch, and an unknown arch is not checked", (t) => {
  const dir = tempDir(t);
  const text = path.join(dir, "notes.txt");
  fs.writeFileSync(text, "not a binary");
  assert.equal(isBuiltForArch(text, "x64"), false);
  assert.equal(isBuiltForArch(path.join(dir, "missing.exe"), "x64"), false);
  assert.equal(isBuiltForArch(text, "riscv64"), true);
});

test("a wrong-arch build set aside comes back only as a last resort", (t) => {
  const exe = path.join(tempDir(t), "helper.exe");
  const logs = [];
  const fallback = createWrongArchFallback(exe, "arm64", (line) => logs.push(line));

  writePe(exe, PE_MACHINE.x64);
  fallback.setAside("Binary built with MinGW-w64");
  assert.equal(fs.existsSync(exe), false);
  assert.match(logs.at(-1), /not built for arm64/);

  assert.equal(fallback.restore(), true);
  assert.equal(readPeMachine(exe), PE_MACHINE.x64);
  assert.match(logs.at(-1), /runs emulated in development/);
  assert.equal(fallback.restore(), false);
});

test("a native build found later drops the set-aside one", (t) => {
  const exe = path.join(tempDir(t), "helper.exe");
  const fallback = createWrongArchFallback(exe, "arm64", () => {});
  writePe(exe, PE_MACHINE.x64);
  fallback.setAside("Downloaded binary");
  writePe(exe, PE_MACHINE.arm64);
  fallback.discard();
  assert.equal(fallback.restore(), false);
  assert.equal(readPeMachine(exe), PE_MACHINE.arm64);
});

test("a set-aside binary left by an interrupted run is cleared on the next one", (t) => {
  const exe = path.join(tempDir(t), "helper.exe");
  writePe(`${exe}.other-arch`, PE_MACHINE.x64);
  const fallback = createWrongArchFallback(exe, "arm64", () => {});
  assert.equal(fs.existsSync(`${exe}.other-arch`), false);
  assert.equal(fallback.restore(), false);
});
