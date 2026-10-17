export type BridgeConfig = {
  host: string;
  port: string;
  flipH: boolean;
  flipV: boolean;
};

export function bridgeFrameUrl(config: BridgeConfig): string {
  const host = config.host.trim();
  const port = config.port.trim();
  return `http://${host}:${port}/frame`;
}

/** Same-origin upload when HTTPS (avoids mixed content + works on iPhone). */
export function getFrameUploadUrl(config: BridgeConfig): string {
  if (typeof window === "undefined") return "/api/frame";
  if (window.isSecureContext) return "/api/frame";
  const host = window.location.hostname;
  if (host === "localhost" || host === "127.0.0.1") return "/api/frame";
  return bridgeFrameUrl(config);
}

export async function uploadFrame(
  config: BridgeConfig,
  jpegBlob: Blob,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(getFrameUploadUrl(config), {
    method: "POST",
    headers: {
      "Content-Type": "image/jpeg",
      "X-Flip-H": config.flipH ? "1" : "0",
      "X-Flip-V": config.flipV ? "1" : "0",
    },
    body: jpegBlob,
    signal,
  });

  if (!response.ok) {
    let message = `Upload failed (${response.status})`;
    try {
      const data = (await response.json()) as { error?: string };
      if (data.error) message = data.error;
    } catch {
      /* non-JSON body */
    }
    throw new Error(message);
  }
}

export const DEFAULT_BRIDGE_HOST = "192.168.0.2";
export const DEFAULT_BRIDGE_PORT = "8080";

export const STORAGE_KEYS = {
  host: "chess-bridge-host",
  port: "chess-bridge-port",
  flipH: "chess-bridge-flip-h",
  flipV: "chess-bridge-flip-v",
} as const;
