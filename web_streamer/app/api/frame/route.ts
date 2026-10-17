import { NextRequest, NextResponse } from "next/server";
import http from "node:http";

const BRIDGE_BASE =
  process.env.BRIDGE_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8080";

/**
 * Post a buffer to the bridge using node:http directly.
 *
 * Why not fetch(): Node's undici fetch on Windows has a long-standing issue
 * forwarding to 127.0.0.1 / localhost where it either hangs or fails with
 * "fetch failed" due to IPv6 ordering. Using node:http with an explicit
 * IPv4 host and family=4 bypasses that resolver path entirely.
 */
function postToBridge(
  target: string,
  body: Buffer,
  headers: Record<string, string>,
): Promise<{ status: number; body: string }> {
  return new Promise((resolve, reject) => {
    const url = new URL(target);
    const req = http.request(
      {
        protocol: url.protocol,
        hostname: url.hostname,
        port: url.port || 80,
        path: url.pathname + url.search,
        method: "POST",
        family: 4,
        headers: {
          ...headers,
          "Content-Length": body.length.toString(),
        },
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (chunk: Buffer) => chunks.push(chunk));
        res.on("end", () => {
          resolve({
            status: res.statusCode ?? 0,
            body: Buffer.concat(chunks).toString("utf-8"),
          });
        });
      },
    );
    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

export async function POST(request: NextRequest) {
  const target = `${BRIDGE_BASE}/frame`;
  const headers: Record<string, string> = {
    "Content-Type": request.headers.get("content-type") ?? "image/jpeg",
  };
  const flipH = request.headers.get("x-flip-h");
  const flipV = request.headers.get("x-flip-v");
  if (flipH) headers["X-Flip-H"] = flipH;
  if (flipV) headers["X-Flip-V"] = flipV;

  const body = Buffer.from(await request.arrayBuffer());
  try {
    const upstream = await postToBridge(target, body, headers);
    if (upstream.status < 200 || upstream.status >= 300) {
      return NextResponse.json(
        { error: `Bridge returned ${upstream.status}: ${upstream.body}` },
        { status: upstream.status || 502 },
      );
    }
    return new NextResponse(null, { status: 204 });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[/api/frame] Bridge connect failed: target=${target} error=${message}`);
    return NextResponse.json(
      {
        error: `Could not reach the Python bridge at ${target}. ${message}`,
      },
      { status: 502 },
    );
  }
}
