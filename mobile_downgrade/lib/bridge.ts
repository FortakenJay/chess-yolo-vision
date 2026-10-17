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

export async function uploadFrameFromPath(
  config: BridgeConfig,
  filePath: string,
): Promise<void> {
  const normalized = filePath.startsWith("file://") ? filePath : `file://${filePath}`;
  const fileResponse = await fetch(normalized);
  const blob = await fileResponse.blob();

  const response = await fetch(bridgeFrameUrl(config), {
    method: "POST",
    headers: {
      "Content-Type": "image/jpeg",
      "X-Flip-H": config.flipH ? "1" : "0",
      "X-Flip-V": config.flipV ? "1" : "0",
    },
    body: blob,
  });

  if (!response.ok) {
    throw new Error(`Bridge returned ${response.status}`);
  }
}
