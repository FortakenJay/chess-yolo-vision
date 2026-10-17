# Chess Phone Camera (Expo Go)

This app replaces ESP32-CAM as the live camera source for `Chess-YOKOv8-vision`.

It captures frames from your phone camera and uploads JPEG frames to the desktop bridge:
- `POST http://<pc-ip>:8080/frame` from phone
- Bridge exposes MJPEG at `http://127.0.0.1:8080/stream`
- Python pipeline reads that MJPEG URL through existing `src/stream.py`

## Run

From `mobile/`:

```bash
npm install
npx expo start
```

Open Expo Go on iPhone and scan the QR code.

## Bridge + pipeline flow

1. Start bridge on PC (repo root):

   ```powershell
   .\.venv\Scripts\python code/phone_stream_bridge.py --host 0.0.0.0 --port 8080
   ```

2. In the phone app settings panel:
   - Host: your PC LAN IP (from `ipconfig`)
   - Port: `8080`
   - Target FPS: `10-15`
   - JPEG quality: `0.70-0.90`
   - Optional flip toggles to match board orientation

3. Tap **Start Streaming**.

4. Verify bridge stream from PC:

   ```powershell
   .\.venv\Scripts\python code/view_esp32_stream.py --url http://127.0.0.1:8080/stream
   ```

5. Ensure `config.yaml` points to bridge:
   - `esp32_url: "http://127.0.0.1:8080/stream"`

6. Run full pipeline:

   ```powershell
   .\.venv\Scripts\python main.py
   ```

## Notes

- Keep the phone app in foreground while streaming.
- Phone and PC must be on the same Wi-Fi network.
- Allow inbound TCP `8080` on Windows firewall (Private network).
