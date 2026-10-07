import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "./ui/button";
import { DownloadProgressBar } from "./ui/DownloadProgressBar";
import { Check, CircleAlert, Zap } from "./icons";
import type { DownloadProgress } from "../hooks/useModelDownload";
import type { NpuWhisperStatus } from "../types/electron";

interface WhisperNpuCardProps {
  /** The selected local Whisper model */
  model: string;
  /** Its CPU (ggml) build is installed: the NPU falls back to it, so it comes first */
  cpuModelDownloaded: boolean;
}

const EMPTY_PROGRESS: DownloadProgress = { percentage: 0, downloadedBytes: 0, totalBytes: 0 };

function formatSize(bytes: number): string {
  return bytes >= 1_000_000_000
    ? `${(bytes / 1_000_000_000).toFixed(1)} GB`
    : `${Math.round(bytes / 1_000_000)} MB`;
}

// Whisper on the Snapdragon NPU (Windows on Arm). Each model has its own NPU
// build, downloaded from Qualcomm AI Hub on top of the CPU model; main runs it
// with whisper-npu-server and falls back to the CPU model if the NPU fails.
export default function WhisperNpuCard({ model, cpuModelDownloaded }: WhisperNpuCardProps) {
  const { t } = useTranslation();
  const [status, setStatus] = useState<NpuWhisperStatus | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [installing, setInstalling] = useState(false);
  const [progress, setProgress] = useState<DownloadProgress>(EMPTY_PROGRESS);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [dismissed, setDismissed] = useState(false);
  // A server reload onto the NPU is in flight
  const [activating, setActivating] = useState(false);
  // Live truth from the running server, never inferred from a download
  const [active, setActive] = useState(false);
  const cancelledRef = useRef(false);

  const readStatus = useCallback(async () => {
    try {
      const next = await window.electronAPI?.getNpuWhisperStatus?.(model);
      setStatus(next ?? null);
      if (next?.downloading) setDownloading(true);
    } catch {
      setStatus(null);
    }
  }, [model]);

  useEffect(() => {
    setDownloadError(null);
    readStatus();
    return window.electronAPI?.onWhisperGpuStatusChanged?.(readStatus);
  }, [readStatus]);

  useEffect(() => {
    return window.electronAPI?.onNpuFallbackNotification?.(() => {
      setActivating(false);
      setActive(false);
      readStatus();
    });
  }, [readStatus]);

  useEffect(() => {
    if (!downloading) return;
    return window.electronAPI?.onNpuWhisperDownloadProgress?.((data) => {
      if (data.model !== model) return;
      setInstalling(data.type === "installing");
      setProgress({
        percentage: data.percentage,
        downloadedBytes: data.downloaded_bytes ?? 0,
        totalBytes: data.total_bytes ?? 0,
      });
    });
  }, [downloading, model]);

  useEffect(() => {
    if (!status?.downloaded) return;
    const poll = () => {
      window.electronAPI
        ?.whisperServerStatus?.()
        .then((server) => {
          const onNpu = !!server?.running && server.gpuBackend === "npu";
          setActive(onNpu);
          if (onNpu) setActivating(false);
        })
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, activating ? 1000 : 5000);
    return () => clearInterval(id);
  }, [status?.downloaded, activating]);

  // The NPU model loads in seconds; past this the live status settles it
  useEffect(() => {
    if (!activating) return;
    const timeout = setTimeout(() => setActivating(false), 60_000);
    return () => clearTimeout(timeout);
  }, [activating]);

  const handleDownload = async () => {
    cancelledRef.current = false;
    setDownloadError(null);
    setProgress(EMPTY_PROGRESS);
    setInstalling(false);
    setDownloading(true);
    try {
      const result = await window.electronAPI?.downloadNpuWhisperModel?.(model);
      if (result?.success) {
        setActivating(!!result.willRestart);
      } else if (result && !cancelledRef.current) {
        setDownloadError(result.error ?? "");
      }
    } finally {
      setDownloading(false);
      setInstalling(false);
      readStatus();
    }
  };

  const handleCancel = async () => {
    cancelledRef.current = true;
    await window.electronAPI?.cancelNpuWhisperDownload?.();
  };

  const handleRetry = async () => {
    const result = await window.electronAPI?.whisperGpuRetry?.();
    setActivating(!!result?.willRestart);
    readStatus();
  };

  const handleRemove = async () => {
    await window.electronAPI?.deleteNpuWhisperModel?.(model);
    setActivating(false);
    setActive(false);
    readStatus();
  };

  if (!status?.supported || !status.npuAvailable || !status.modelSupported || !cpuModelDownloaded) {
    return null;
  }

  if (downloading) {
    return (
      <div>
        <DownloadProgressBar
          modelName={t("npu.downloadLabel")}
          progress={progress}
          isInstalling={installing}
        />
        {!installing && (
          <div className="px-2.5 pb-1 flex justify-end">
            <button
              onClick={handleCancel}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {t("gpu.cancel")}
            </button>
          </div>
        )}
      </div>
    );
  }

  if (!status.downloaded && dismissed) return null;

  const failed = !!status.downloaded && !!status.gpuFailed;

  return (
    <div
      className={`rounded-md border p-2.5 ${
        failed ? "border-warning/40 bg-warning/5" : "border-border bg-surface-1"
      }`}
    >
      {failed ? (
        <div className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 items-start gap-2">
            <CircleAlert size={15} className="mt-0.5 shrink-0 text-warning" />
            <div className="min-w-0">
              <p className="text-xs font-medium text-foreground">{t("npu.activationFailed")}</p>
              <p className="mt-0.5 text-xs leading-snug text-muted-foreground">
                {t("npu.activationFailedDescription")}
              </p>
              {status.gpuFailReason && (
                <p
                  dir="ltr"
                  className="mt-1 wrap-break-word font-mono text-[11px] leading-snug text-muted-foreground"
                >
                  {status.gpuFailReason}
                </p>
              )}
              <Button onClick={handleRetry} size="sm" className="mt-2 h-7 px-3 text-xs">
                {t("npu.retryActivation")}
              </Button>
            </div>
          </div>
          <Button
            onClick={handleRemove}
            size="sm"
            variant="ghost"
            className="h-6 shrink-0 px-2 text-xs text-muted-foreground hover:text-destructive"
          >
            {t("gpu.remove")}
          </Button>
        </div>
      ) : status.downloaded ? (
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            {activating ? (
              <>
                <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-primary animate-pulse" />
                <span className="text-xs font-medium text-foreground">{t("npu.activating")}</span>
              </>
            ) : active ? (
              <>
                <Check size={13} className="text-success" />
                <span className="text-xs font-medium text-foreground">{t("npu.active")}</span>
              </>
            ) : (
              <>
                <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0 bg-primary" />
                <span className="text-xs font-medium text-foreground">{t("npu.ready")}</span>
              </>
            )}
          </div>
          <Button
            onClick={handleRemove}
            size="sm"
            variant="ghost"
            className="h-6 px-2 text-xs text-muted-foreground hover:text-destructive"
          >
            {t("gpu.remove")}
          </Button>
        </div>
      ) : (
        <div className="flex items-start gap-2.5">
          <Zap size={13} className="text-primary shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium text-foreground">{t("npu.banner")}</p>
            <p className="mt-0.5 text-xs leading-snug text-muted-foreground">
              {t("npu.bannerDescription", { size: formatSize(status.downloadSizeBytes ?? 0) })}
            </p>
            <div className="flex items-center gap-2 mt-1.5">
              <Button
                onClick={handleDownload}
                size="sm"
                variant="default"
                className="h-6 px-2.5 text-xs"
              >
                {t("npu.enableButton")}
              </Button>
              <button
                onClick={() => setDismissed(true)}
                className="text-xs text-muted-foreground hover:text-foreground transition-colors"
              >
                {t("gpu.dismiss")}
              </button>
            </div>
          </div>
        </div>
      )}
      {!status.downloaded && downloadError !== null && (
        <p className="mt-2 text-xs leading-snug text-destructive">
          {t("npu.downloadFailed")}
          {downloadError && ` ${downloadError}`}
        </p>
      )}
    </div>
  );
}
