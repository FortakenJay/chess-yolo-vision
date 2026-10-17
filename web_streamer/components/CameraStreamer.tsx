"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  type BridgeConfig,
  DEFAULT_BRIDGE_HOST,
  DEFAULT_BRIDGE_PORT,
  STORAGE_KEYS,
  uploadFrame,
} from "@/lib/bridge";
import { cameraBlockedMessage, cameraStatus, canUseCamera } from "@/lib/camera";

function readStored(key: string, fallback: string): string {
  return window.localStorage.getItem(key) ?? fallback;
}

function readStoredBool(key: string, fallback: boolean): boolean {
  const raw = window.localStorage.getItem(key);
  if (raw === null) return fallback;
  return raw === "1";
}

async function canvasToJpegBlob(
  canvas: HTMLCanvasElement,
  quality = 0.6,
): Promise<Blob | null> {
  // Older iOS Safari can intermittently return null from toBlob.
  const blob = await new Promise<Blob | null>((resolve) => {
    canvas.toBlob(resolve, "image/jpeg", quality);
  });
  if (blob) return blob;

  // Fallback path for legacy Safari: dataURL -> Blob.
  try {
    const dataUrl = canvas.toDataURL("image/jpeg", quality);
    const response = await fetch(dataUrl);
    return await response.blob();
  } catch {
    return null;
  }
}

export default function CameraStreamer() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const uploadingRef = useRef(false);
  const isStreamingRef = useRef(false);

  const [host, setHost] = useState(DEFAULT_BRIDGE_HOST);
  const [port, setPort] = useState(DEFAULT_BRIDGE_PORT);
  const [flipH, setFlipH] = useState(false);
  const [flipV, setFlipV] = useState(false);
  const [prefsReady, setPrefsReady] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [statusText, setStatusText] = useState("Ready");
  const [senderFps, setSenderFps] = useState(0);
  const [lastError, setLastError] = useState<string | null>(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraAllowed, setCameraAllowed] = useState(true);
  const [cameraBlockReason, setCameraBlockReason] = useState<"insecure-context" | "no-api" | null>(null);

  const sentTimestampsRef = useRef<number[]>([]);
  const configRef = useRef<BridgeConfig>({ host, port, flipH, flipV });

  useEffect(() => {
    setHost(readStored(STORAGE_KEYS.host, DEFAULT_BRIDGE_HOST));
    setPort(readStored(STORAGE_KEYS.port, DEFAULT_BRIDGE_PORT));
    setFlipH(readStoredBool(STORAGE_KEYS.flipH, false));
    setFlipV(readStoredBool(STORAGE_KEYS.flipV, false));
    setPrefsReady(true);

    const status = cameraStatus();
    const allowed = status.ok;
    setCameraAllowed(allowed);
    if (!allowed) {
      setCameraBlockReason(status.reason);
      setLastError(cameraBlockedMessage());
      setStatusText(status.reason === "insecure-context" ? "HTTPS required" : "Camera unavailable");
    }
  }, []);

  useEffect(() => {
    configRef.current = { host, port, flipH, flipV };
    if (!prefsReady) return;
    window.localStorage.setItem(STORAGE_KEYS.host, host);
    window.localStorage.setItem(STORAGE_KEYS.port, port);
    window.localStorage.setItem(STORAGE_KEYS.flipH, flipH ? "1" : "0");
    window.localStorage.setItem(STORAGE_KEYS.flipV, flipV ? "1" : "0");
  }, [host, port, flipH, flipV, prefsReady]);

  useEffect(() => {
    isStreamingRef.current = isStreaming;
  }, [isStreaming]);

  const canStream = cameraAllowed;

  const stopMedia = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    mediaStreamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraReady(false);
  }, []);

  const startMedia = useCallback(async () => {
    if (!canUseCamera()) {
      throw new Error(cameraBlockedMessage());
    }

    let stream: MediaStream;
    try {
      // Prefer lighter constraints first; older phones (iPhone 7) are more stable here.
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 10, max: 15 },
        },
        audio: false,
      });
    } catch {
      // Last-resort fallback for strict legacy Safari implementations.
      stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
    }

    mediaStreamRef.current = stream;
    const video = videoRef.current;
    if (!video) {
      throw new Error("Video element missing");
    }

    video.srcObject = stream;
    video.playsInline = true;
    video.muted = true;
    video.autoplay = true;
    (video as HTMLVideoElement & { setAttribute: typeof video.setAttribute }).setAttribute(
      "webkit-playsinline",
      "true",
    );

    await video.play().catch(() => {});

    await new Promise<void>((resolve) => {
      const done = () => resolve();
      if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
        done();
        return;
      }
      const onAny = () => {
        if (video.videoWidth > 0 && video.videoHeight > 0) {
          video.removeEventListener("playing", onAny);
          video.removeEventListener("loadeddata", onAny);
          video.removeEventListener("loadedmetadata", onAny);
          done();
        }
      };
      video.addEventListener("playing", onAny);
      video.addEventListener("loadeddata", onAny);
      video.addEventListener("loadedmetadata", onAny);
    });

    setCameraReady(true);
  }, []);

  const captureAndUpload = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      setStatusText(`Waiting for video data (readyState=${video.readyState})…`);
      return;
    }

    const w = video.videoWidth;
    const h = video.videoHeight;
    if (w === 0 || h === 0) {
      setStatusText("Waiting for camera frame…");
      return;
    }

    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, w, h);

    const blob = await canvasToJpegBlob(canvas, 0.6);
    if (!blob) return;

    await uploadFrame(configRef.current, blob);
    const now = performance.now();
    sentTimestampsRef.current.push(now);
    sentTimestampsRef.current = sentTimestampsRef.current.filter(
      (t) => now - t <= 2000,
    );
    setSenderFps(sentTimestampsRef.current.length / 2);
    setStatusText(`Streaming · ${w}×${h}`);
    setLastError(null);
  }, []);

  const pumpFrames = useCallback(() => {
    const tick = async () => {
      if (!isStreamingRef.current) return;

      if (!uploadingRef.current) {
        uploadingRef.current = true;
        try {
          await captureAndUpload();
        } catch (err) {
          setLastError(err instanceof Error ? err.message : "Upload failed");
          setStatusText("Stream error");
        } finally {
          uploadingRef.current = false;
        }
      }

      rafRef.current = requestAnimationFrame(() => {
        void tick();
      });
    };

    void tick();
  }, [captureAndUpload]);

  const startStreaming = useCallback(async () => {
    if (!canStream) {
      setStatusText("HTTPS required for camera");
      return;
    }
    try {
      setLastError(null);
      setStatusText("Starting camera...");
      await startMedia();
      isStreamingRef.current = true;
      setIsStreaming(true);
      setStatusText("Waiting for first frame…");
      sentTimestampsRef.current = [];
      pumpFrames();
    } catch (err) {
      setLastError(err instanceof Error ? err.message : "Camera start failed");
      setStatusText("Camera error");
      stopMedia();
      setIsStreaming(false);
    }
  }, [canStream, pumpFrames, startMedia, stopMedia]);

  const stopStreaming = useCallback(() => {
    isStreamingRef.current = false;
    setIsStreaming(false);
    setStatusText("Stopped");
    setSenderFps(0);
    stopMedia();
  }, [stopMedia]);

  useEffect(() => {
    const onVisibility = () => {
      if (document.hidden && isStreamingRef.current) {
        setIsStreaming(false);
        stopMedia();
        setStatusText("Paused (tab hidden)");
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, [stopMedia]);

  useEffect(() => () => stopMedia(), [stopMedia]);

  return (
    <Shell>
      <video
        ref={videoRef}
        className="absolute inset-0 h-full w-full object-cover"
        autoPlay
        playsInline
        muted
      />
      <canvas ref={canvasRef} className="hidden" aria-hidden />

      <TopBar
        senderFps={senderFps}
        statusText={statusText}
        onOpenSettings={() => setShowSettings(true)}
      />

      <BottomBar
        flipH={flipH}
        flipV={flipV}
        isStreaming={isStreaming}
        canStream={canStream}
        lastError={lastError}
        onFlipH={() => setFlipH((v) => !v)}
        onFlipV={() => setFlipV((v) => !v)}
        onToggle={() => (isStreaming ? stopStreaming() : void startStreaming())}
      />

      {showSettings && (
        <SettingsModal
          host={host}
          port={port}
          onHost={setHost}
          onPort={setPort}
          onClose={() => setShowSettings(false)}
        />
      )}

      {!cameraReady && !isStreaming && cameraAllowed && (
        <p className="pointer-events-none absolute inset-x-0 top-1/2 -translate-y-1/2 text-center text-sm text-white/80">
          Tap Start Streaming — allow camera access
        </p>
      )}

      {cameraBlockReason && <CameraBlockedOverlay reason={cameraBlockReason} />}
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative h-[100dvh] w-full overflow-hidden bg-black">
      {children}
    </div>
  );
}

function TopBar({
  senderFps,
  statusText,
  onOpenSettings,
}: {
  senderFps: number;
  statusText: string;
  onOpenSettings: () => void;
}) {
  return (
    <div className="absolute left-0 right-0 top-0 z-20 flex items-start justify-between p-3 pt-[max(12px,env(safe-area-inset-top))]">
      <div className="rounded-xl bg-black/55 px-3 py-2">
        <p className="text-xl font-bold text-green-400">
          {senderFps.toFixed(1)} fps
        </p>
        <p className="text-xs text-white/85">{statusText}</p>
      </div>
      <button
        type="button"
        onClick={onOpenSettings}
        className="flex h-11 w-11 items-center justify-center rounded-full bg-black/55 text-white"
        aria-label="Settings"
      >
        ⚙
      </button>
    </div>
  );
}

function BottomBar({
  flipH,
  flipV,
  isStreaming,
  canStream,
  lastError,
  onFlipH,
  onFlipV,
  onToggle,
}: {
  flipH: boolean;
  flipV: boolean;
  isStreaming: boolean;
  canStream: boolean;
  lastError: string | null;
  onFlipH: () => void;
  onFlipV: () => void;
  onToggle: () => void;
}) {
  return (
    <div className="absolute bottom-0 left-0 right-0 z-20 p-3 pb-[max(12px,env(safe-area-inset-bottom))]">
      <div className="rounded-2xl border border-white/10 bg-black/70 p-4 backdrop-blur-md">
        <div className="grid grid-cols-2 gap-2">
          <FlipButton label="Flip H" on={flipH} onClick={onFlipH} />
          <FlipButton label="Flip V" on={flipV} onClick={onFlipV} />
        </div>
        <button
          type="button"
          disabled={!canStream}
          onClick={onToggle}
          className={`mt-3 w-full rounded-xl py-4 text-lg font-bold text-white ${
            isStreaming
              ? "bg-red-600/90"
              : "bg-emerald-600/90 disabled:bg-zinc-600"
          }`}
        >
          {isStreaming ? "Stop" : "Start Streaming"}
        </button>
        {lastError ? (
          <p className="mt-2 text-xs text-red-400">{lastError}</p>
        ) : null}
      </div>
    </div>
  );
}

function FlipButton({
  label,
  on,
  onClick,
}: {
  label: string;
  on: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-full py-2.5 text-sm font-semibold ${
        on ? "bg-emerald-700/80 text-white" : "bg-white/15 text-white"
      }`}
    >
      {label}: {on ? "On" : "Off"}
    </button>
  );
}

function SettingsModal({
  host,
  port,
  onHost,
  onPort,
  onClose,
}: {
  host: string;
  port: string;
  onHost: (v: string) => void;
  onPort: (v: string) => void;
  onClose: () => void;
}) {
  return (
    <SettingsOverlay onClose={onClose}>
      <h2 className="mb-4 text-lg font-semibold">Bridge settings</h2>
      <label className="mb-3 block text-sm text-zinc-400">
        PC LAN IP
        <input
          className="mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-3 text-base text-white"
          value={host}
          onChange={(e) => onHost(e.target.value)}
          inputMode="url"
          autoCapitalize="none"
          autoCorrect="off"
          autoComplete="off"
          spellCheck={false}
          placeholder="192.168.0.2"
        />
      </label>
      <label className="mb-4 block text-sm text-zinc-400">
        Port
        <input
          className="mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-800 px-3 py-3 text-base text-white"
          value={port}
          onChange={(e) => onPort(e.target.value)}
          inputMode="numeric"
          placeholder="8080"
        />
      </label>
      <p className="mb-4 text-xs text-zinc-500">
        Only needed on plain HTTP. When using the Cloudflare tunnel URL
        (npm run dev:phone) frames are proxied automatically — this field is ignored.
      </p>
      <button
        type="button"
        onClick={onClose}
        className="w-full rounded-xl bg-blue-600 py-3 font-semibold"
      >
        Done
      </button>
    </SettingsOverlay>
  );
}

function CameraBlockedOverlay({ reason }: { reason: "insecure-context" | "no-api" }) {
  return (
    <div className="absolute inset-0 z-40 flex flex-col items-center justify-center gap-5 bg-black/92 px-6 text-center">
      <p className="text-4xl">🔒</p>
      {reason === "insecure-context" ? (
        <>
          <h2 className="text-lg font-bold text-red-400">Camera blocked — HTTP not allowed</h2>
          <p className="text-sm leading-relaxed text-white/80">
            Safari blocks camera access on plain <span className="font-mono text-yellow-300">http://</span> addresses.
          </p>
          <div className="w-full rounded-xl border border-white/10 bg-zinc-900 p-4 text-left text-xs text-white/70">
            <p className="mb-2 font-semibold text-white">Fix — on your PC:</p>
            <p className="mb-1 font-mono text-green-400">cd web_streamer</p>
            <p className="font-mono text-green-400">npm run dev:phone</p>
            <p className="mt-3 mb-1 text-white/50">Wait for the line:</p>
            <p className="font-mono text-sky-300">https://xxxx.trycloudflare.com</p>
            <p className="mt-3 text-white/50">Then open THAT URL on iPhone Safari.</p>
          </div>
        </>
      ) : (
        <>
          <h2 className="text-lg font-bold text-red-400">Camera API unavailable</h2>
          <p className="text-sm leading-relaxed text-white/80">
            Try Safari. Other browsers may not support camera capture.
          </p>
        </>
      )}
    </div>
  );
}

function SettingsOverlay({
  children,
  onClose,
}: {
  children: React.ReactNode;
  onClose: () => void;
}) {
  return (
    <div
      className="absolute inset-0 z-30 flex items-end bg-black/50"
      role="presentation"
      onClick={onClose}
    >
      <div
        className="w-full rounded-t-2xl bg-zinc-900 p-5 text-white pb-[max(20px,env(safe-area-inset-bottom))]"
        role="dialog"
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>
  );
}
