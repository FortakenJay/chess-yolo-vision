# Chess Phone Stream (Safari / Next.js)

Mobile-first web app for iPhone Safari that streams live camera frames to the desktop bridge (no Expo, no AltStore).

Uses the browser **MediaDevices API** (`getUserMedia`) — the same stack Safari uses for WebRTC camera capture. Frames are drawn to a canvas and posted as JPEG to your existing Python bridge.

## Flow

```text
iPhone Safari (this app)  --POST /frame-->  phone_stream_bridge.py
                                                      |
                                                      v
                                            GET /stream (MJPEG)
                                                      |
                                                      v
                                            main.py (OpenCV pipeline)
```

## Run

### 1) Bridge on Windows (repo root)

```powershell
.\venv\Scripts\python.exe code/phone_stream_bridge.py --host 0.0.0.0 --port 8080
```

### 2) Web app on PC

```powershell
cd web_streamer
npm install
npm run dev
```

Dev server binds `0.0.0.0:3000` so your phone can reach it on LAN.

### 3) iPhone Safari

1. Same Wi-Fi as PC (not cellular, not guest Wi-Fi with client isolation).
2. On Windows run `ipconfig` and find **IPv4** (example: `192.168.0.2`).
3. Open **`http://192.168.0.2:3000`** on iPhone (use your real IP, not `0.0.0.0` or `localhost`).

**If the page won't load or is broken:**

1. **Restart** `npm run dev` after any `next.config.ts` change.
2. **Windows Firewall** — allow inbound TCP port 3000 (Private network):

   ```powershell
   netsh advfirewall firewall add rule name="Next.js dev 3000" dir=in action=allow protocol=TCP localport=3000
   ```

3. **Next.js 16 LAN block** — if the terminal shows `Blocked cross-origin request ... allowedDevOrigins`, add your PC IP to `allowedDevOrigins` in `next.config.ts` (already set for `192.168.0.2` in this repo).
3. Tap **Settings** (gear) → set **PC LAN IP** and port `8080`.
4. Tap **Start Streaming** and allow camera access.
5. Keep Safari in the foreground.

### 4) Verify pipeline

```powershell
.\venv\Scripts\python.exe code/view_esp32_stream.py --url http://127.0.0.1:8080/stream
.\venv\Scripts\python.exe main.py
```

## Notes

- **HTTPS:** Safari may require a secure context for camera on some networks. If camera is blocked over `http://`, use HTTPS (e.g. tunnel) or test on `localhost` during development.
- **CORS:** Bridge enables browser uploads from the Next app origin.
- **No install:** Add to Home Screen in Safari for a full-screen shortcut (optional).
