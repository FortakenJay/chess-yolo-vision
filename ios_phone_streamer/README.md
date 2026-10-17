# Native iPhone Streamer (AltStore)

This folder contains a native Swift app that streams live camera video frames to the desktop bridge:

- Phone -> `POST http://<pc-ip>:8080/frame`
- Bridge -> `GET http://127.0.0.1:8080/stream`
- Python pipeline reads `esp32_url` from `config.yaml`

## Why this app

Expo Go captures snapshots (`takePictureAsync`) instead of true camera frame callbacks.
This app uses `AVCaptureVideoDataOutput` for real frame streaming.

## Project generation (on Mac)

This repo ships an `XcodeGen` spec (`project.yml`) so you can generate the Xcode project:

```bash
brew install xcodegen
cd ios_phone_streamer
xcodegen generate
open PhoneStreamer.xcodeproj
```

## Build + sideload via AltStore

1. Open `PhoneStreamer.xcodeproj` in Xcode.
2. Set your Apple team in Signing & Capabilities.
3. Build once for device to verify permissions/network access.
4. Archive app and export `.ipa`.
5. Install `.ipa` through AltStore.

## GitHub Actions (build IPA without daily Mac use)

Workflow file: `.github/workflows/ios-phone-streamer.yml`

### One-time setup

1. Push this repo to GitHub.
2. You still need **signing material** from Apple once (free Apple ID works; paid Developer is easier):
   - Use your 2012 Mac (or any Mac) with Xcode once, **or** a paid Apple Developer account.
3. In Xcode on Mac:
   - Open generated `PhoneStreamer.xcodeproj`.
   - Set your Team under Signing.
   - Plug in iPhone once and run on device (creates development cert + profile).
4. Export secrets for GitHub (repo → **Settings → Secrets and variables → Actions**):

| Secret | What it is |
|--------|------------|
| `IOS_CERTIFICATE_P12` | Base64 of your `.p12` signing certificate |
| `IOS_CERTIFICATE_PASSWORD` | Password for that `.p12` |
| `IOS_PROVISION_PROFILE_BASE64` | Base64 of `.mobileprovision` for `com.jay.phonestreamer` |
| `IOS_DEVELOPMENT_TEAM` | 10-character Team ID (Apple Developer → Membership) |
| `IOS_CODE_SIGN_IDENTITY` | e.g. `Apple Development: Your Name (XXXXXXXXXX)` |
| `IOS_PROVISIONING_PROFILE_NAME` | Exact profile name in Xcode |

**Create `.p12` on Mac (Keychain Access):**

- Keychain → My Certificates → right-click **Apple Development: …** → Export → `.p12`.
- Base64 for GitHub (Terminal):

  ```bash
  base64 -i cert.p12 | pbcopy
  ```

- Profile file:

  ```bash
  base64 -i YourProfile.mobileprovision | pbcopy
  ```

### Run the workflow

1. GitHub → **Actions** → **iOS PhoneStreamer IPA** → **Run workflow**.
2. Wait for green check (~5–10 min).
3. Open the run → **Artifacts** → download `PhoneStreamer-ipa`.
4. Copy `.ipa` to iPhone → **AltStore** → **+** → install.
5. Refresh in AltStore every 7 days (free Apple ID).

### If you have no Mac at all

- Use **`web_streamer/`** (Safari, no build), **or**
- Borrow any Mac once only to create the signing secrets above, **or**
- Use paid Apple Developer + API key workflows (more setup).

## Runtime flow

1. On Windows PC (repo root):

   ```powershell
   .\venv\Scripts\python.exe code/phone_stream_bridge.py --host 0.0.0.0 --port 8080
   ```

2. Open app on iPhone.
3. Enter host (`PC LAN IP`) and port (`8080`).
4. Tap **Start Streaming**.
5. Verify bridge stream on PC:

   ```powershell
   .\venv\Scripts\python.exe code/view_esp32_stream.py --url http://127.0.0.1:8080/stream
   ```

6. Run pipeline:

   ```powershell
   .\venv\Scripts\python.exe main.py
   ```

## Notes

- Keep app foreground while streaming.
- Local network and camera permissions are required.
- App sends flip flags via `X-Flip-H` / `X-Flip-V` headers, handled by bridge.
