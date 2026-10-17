export function isLocalHostname(hostname: string): boolean {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "[::1]";
}

/** Camera works on https:// or http://localhost only (not http://LAN-IP). */
export function canUseCamera(): boolean {
  if (typeof window === "undefined") return false;
  if (!navigator.mediaDevices?.getUserMedia) return false;
  return window.isSecureContext || isLocalHostname(window.location.hostname);
}

export type CameraStatus =
  | { ok: true }
  | { ok: false; reason: "insecure-context"; url: string }
  | { ok: false; reason: "no-api" };

export function cameraStatus(): CameraStatus {
  if (typeof window === "undefined") return { ok: false, reason: "no-api" };
  if (window.isSecureContext || isLocalHostname(window.location.hostname)) {
    if (!navigator.mediaDevices?.getUserMedia) return { ok: false, reason: "no-api" };
    return { ok: true };
  }
  return { ok: false, reason: "insecure-context", url: window.location.href };
}

export function cameraBlockedMessage(): string {
  const s = cameraStatus();
  if (s.ok) return "";
  if (s.reason === "insecure-context") {
    return "Camera blocked: page must be loaded over HTTPS. Run npm run dev:phone on the PC and use the printed https://…trycloudflare.com URL on iPhone.";
  }
  return "Camera API not available in this browser";
}
