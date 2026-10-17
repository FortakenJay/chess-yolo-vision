# Chess Phone Cam (Expo Dev Client + Vision Camera)

SDK 54 app with **expo-dev-client** and **react-native-vision-camera** for faster live frame capture than Expo Go (`takePictureAsync`).

Streams JPEG frames to the desktop bridge:

- Phone → `POST http://<pc-ip>:8080/frame`
- Bridge → `GET http://127.0.0.1:8080/stream`
- Pipeline → `main.py`

## Important: not Expo Go

This app **does not run in Expo Go**. You need a **development build** (custom dev client).

## Build dev client (no Mac — EAS cloud)

1. Install EAS CLI and log in:

   ```bash
   npm install -g eas-cli
   eas login
   ```

2. From this folder:

   ```bash
   cd mobile_downgrade
   eas build:configure
   eas build --profile development --platform ios
   ```

3. When the build finishes, open the install link on your iPhone (or download `.ipa` for AltStore).

4. Start Metro for the dev client:

   ```bash
   npm install
   npm start
   ```

5. Open the **Chess Phone Cam Downgrade** app (not Expo Go) and load the bundle.

## Run with pipeline

**PC — bridge:**

```powershell
.\venv\Scripts\python.exe code/phone_stream_bridge.py --host 0.0.0.0 --port 8080
```

**Phone — app:**

- Settings (gear) → PC LAN IP + port `8080`
- **Start Streaming**

**PC — verify:**

```powershell
.\venv\Scripts\python.exe code/view_esp32_stream.py --url http://127.0.0.1:8080/stream
.\venv\Scripts\python.exe main.py
```

## No-build alternative

Use `web_streamer/` in iPhone Safari — no EAS, no AltStore.

## Android

```bash
eas build --profile development --platform android
```

Install APK, then `npm start` and open the dev client app.
