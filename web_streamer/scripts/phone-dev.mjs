/**
 * Starts Next.js + a Cloudflare quick tunnel so iPhone Safari gets real HTTPS
 * (no self-signed cert trust required).
 */
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const isWin = process.platform === "win32";

function run(cmd, args, opts = {}) {
  return spawn(cmd, args, {
    cwd: root,
    stdio: opts.stdio ?? "inherit",
    shell: isWin,
    env: process.env,
    ...opts,
  });
}

console.log("");
console.log("Starting Next.js on http://127.0.0.1:3000 ...");
console.log("(Keep phone_stream_bridge.py running on port 8080)");
console.log("");

const next = run("npx", ["next", "dev", "-H", "127.0.0.1", "-p", "3000"], {
  stdio: "inherit",
});

let tunnelUrl = null;

const tunnel = run(
  "npx",
  ["--yes", "cloudflared@latest", "tunnel", "--url", "http://127.0.0.1:3000"],
  { stdio: ["ignore", "pipe", "pipe"] },
);

function onTunnelLine(chunk) {
  const text = chunk.toString();
  process.stderr.write(text);
  const match = text.match(/https:\/\/[a-z0-9-]+\.trycloudflare\.com/i);
  if (match && match[0] !== tunnelUrl) {
    tunnelUrl = match[0];
    console.log("");
    console.log("============================================================");
    console.log("  iPhone Safari — open this URL (camera will work):");
    console.log(`  ${tunnelUrl}`);
    console.log("============================================================");
    console.log("");
  }
}

tunnel.stdout.on("data", onTunnelLine);
tunnel.stderr.on("data", onTunnelLine);

function shutdown() {
  next.kill("SIGTERM");
  tunnel.kill("SIGTERM");
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

next.on("exit", (code) => {
  tunnel.kill("SIGTERM");
  process.exit(code ?? 0);
});

tunnel.on("exit", (code) => {
  if (code && code !== 0) {
    console.error("");
    console.error("Cloudflare tunnel failed. Install manually:");
    console.error("  winget install Cloudflare.cloudflared");
    console.error("Or use LAN HTTPS: npm run firewall:open && npm run dev:https");
    console.error("");
  }
});
