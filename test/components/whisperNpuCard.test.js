const test = require("node:test");
const assert = require("node:assert/strict");
const React = require("react");
const { createRoot } = require("react-dom/client");

const { createElement } = React;
const { createRendererServer, installBrowserGlobals } = require("../lib/rendererTestHarness");
const { installInteractiveDom, findElement } = require("../lib/interactiveDom");

// Labels render as their keys; the progress bar renders as a marker.
const MOCKS = {
  "react-i18next": `
    const t = (key) => key;
    export const useTranslation = () => ({ t, i18n: { language: "en" } });
  `,
  "/ui/DownloadProgressBar": `
    import React from "react";
    export const DownloadProgressBar = ({ progress }) =>
      React.createElement("div", null, "PROGRESS " + progress.percentage + "%");
  `,
};

function click(element) {
  element.dispatchEvent({
    type: "click",
    bubbles: true,
    button: 0,
    defaultPrevented: false,
    cancelBubble: false,
    preventDefault() {
      this.defaultPrevented = true;
    },
    stopPropagation() {
      this.cancelBubble = true;
    },
  });
}

function button(root, label) {
  const found = findElement(
    root,
    (element) => element.tagName === "BUTTON" && element.textContent === label
  );
  assert.ok(found, `button ${label} is rendered`);
  return found;
}

// What main reports for a model: an NPU build exists, not downloaded yet
function npuStatus(model, overrides = {}) {
  return {
    supported: true,
    npuAvailable: true,
    model,
    modelSupported: true,
    downloaded: false,
    downloadSizeBytes: 573_000_000,
    downloading: false,
    downloadProgress: 0,
    ...overrides,
  };
}

async function renderCard(t, electronAPI) {
  let root = null;
  t.after(async () => {
    if (root) await React.act(async () => root.unmount());
  });
  installBrowserGlobals(t, { window: { electronAPI } });
  const container = installInteractiveDom(t);
  const vite = await createRendererServer(t, {
    cachePrefix: "openwhispr-whisper-npu-card-test-",
    noExternal: ["react-i18next"],
    mockModules: MOCKS,
  });
  const { default: WhisperNpuCard } = await vite.ssrLoadModule("/components/WhisperNpuCard.tsx");
  root = createRoot(container);
  const render = (model) =>
    React.act(async () =>
      root.render(createElement(WhisperNpuCard, { model, cpuModelDownloaded: true }))
    );
  const settle = (ms = 0) =>
    React.act(async () => {
      await new Promise((resolve) => setTimeout(resolve, ms));
    });
  return { container, render, settle };
}

const baseApi = (overrides) => ({
  onWhisperGpuStatusChanged: () => () => {},
  onNpuFallbackNotification: () => () => {},
  onNpuWhisperDownloadProgress: () => () => {},
  whisperServerStatus: async () => ({ running: false }),
  ...overrides,
});

// Settings reopened (or another picker mounted) while a download runs: no
// download call here will report its end, and a failure broadcasts nothing.
test("a download that was running before the card mounted ends when the status says so", async (t) => {
  let status = npuStatus("small", { downloading: true, downloadProgress: 40 });
  const { container, render, settle } = await renderCard(
    t,
    baseApi({ getNpuWhisperStatus: async () => status })
  );
  await render("small");
  await settle();
  assert.match(container.textContent, /PROGRESS 40%/);

  status = npuStatus("small", { downloaded: true });
  await settle(1300);
  assert.doesNotMatch(container.textContent, /PROGRESS/);
  assert.match(container.textContent, /npu\.ready/);
});

test("a resumed download that fails returns to the offer instead of a frozen bar", async (t) => {
  let status = npuStatus("small", { downloading: true, downloadProgress: 70 });
  const { container, render, settle } = await renderCard(
    t,
    baseApi({ getNpuWhisperStatus: async () => status })
  );
  await render("small");
  await settle();
  assert.match(container.textContent, /PROGRESS 70%/);

  status = npuStatus("small");
  await settle(1300);
  assert.doesNotMatch(container.textContent, /PROGRESS/);
  assert.match(container.textContent, /npu\.enableButton/);
});

// The card stays mounted when the selected model changes, so a download
// started for one model must neither show under nor describe another.
test("switching models during a download shows the new model, not the old download", async (t) => {
  const downloaded = new Set();
  let finishDownload;
  const { container, render, settle } = await renderCard(
    t,
    baseApi({
      getNpuWhisperStatus: async (model) =>
        npuStatus(model, { downloaded: downloaded.has(model), downloading: false }),
      downloadNpuWhisperModel: (model) =>
        new Promise((resolve) => {
          finishDownload = () => {
            downloaded.add(model);
            resolve({ success: true, willRestart: false });
          };
        }),
    })
  );
  await render("small");
  await settle();
  await React.act(async () => click(button(container, "npu.enableButton")));
  assert.match(container.textContent, /PROGRESS 0%/);

  await render("base");
  await settle();
  assert.doesNotMatch(container.textContent, /PROGRESS/);
  assert.match(container.textContent, /npu\.enableButton/);

  // small finishes: its last status read must not land on base
  await React.act(async () => finishDownload());
  await settle();
  assert.doesNotMatch(container.textContent, /npu\.ready/);
  assert.match(container.textContent, /npu\.enableButton/);

  await render("small");
  await settle();
  assert.match(container.textContent, /npu\.ready/);
});

test("the card's own download shows progress until the download call returns", async (t) => {
  const downloaded = new Set();
  let finishDownload;
  const { container, render, settle } = await renderCard(
    t,
    baseApi({
      getNpuWhisperStatus: async (model) =>
        npuStatus(model, { downloaded: downloaded.has(model), downloading: !!finishDownload }),
      downloadNpuWhisperModel: (model) =>
        new Promise((resolve) => {
          finishDownload = () => {
            finishDownload = null;
            downloaded.add(model);
            resolve({ success: true, willRestart: false });
          };
        }),
    })
  );
  await render("small");
  await settle();
  await React.act(async () => click(button(container, "npu.enableButton")));
  await settle(1300);
  assert.match(container.textContent, /PROGRESS/);

  await React.act(async () => finishDownload());
  await settle();
  assert.doesNotMatch(container.textContent, /PROGRESS/);
  assert.match(container.textContent, /npu\.ready/);
});

test("a download that fails while another model is shown reports it on return", async (t) => {
  let failDownload;
  const { container, render, settle } = await renderCard(
    t,
    baseApi({
      getNpuWhisperStatus: async (model) => npuStatus(model),
      downloadNpuWhisperModel: () =>
        new Promise((resolve) => {
          failDownload = () => resolve({ success: false, error: "Not enough disk space" });
        }),
    })
  );
  await render("small");
  await settle();
  await React.act(async () => click(button(container, "npu.enableButton")));
  await render("base");
  await settle();

  await React.act(async () => failDownload());
  await settle();
  assert.doesNotMatch(container.textContent, /npu\.downloadFailed/);

  await render("small");
  await settle();
  assert.match(container.textContent, /npu\.downloadFailed Not enough disk space/);
});
