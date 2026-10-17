import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow iPhone/other devices on LAN to load dev assets (HMR, chunks).
  // Add your PC LAN IP from `ipconfig` if different from 192.168.0.2.
  allowedDevOrigins: [
    "192.168.0.2",
    "192.168.0.2:3000",
    "https://192.168.0.2:3000",
    "*.trycloudflare.com",
    "localhost",
    "localhost:3000",
    "https://localhost:3000",
    "127.0.0.1",
    "127.0.0.1:3000",
    "https://127.0.0.1:3000",
  ],
};

export default nextConfig;
