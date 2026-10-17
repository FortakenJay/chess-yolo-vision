# Hardware

## The camera board

**Freenove ESP32-WROVER CAM**: a classic ESP32 with 4 MB of usable PSRAM and an OV2640 camera.
It shows up on USB as a CH340 serial port (it was `COM8` on this PC).

The firmware is Espressif's `CameraWebServer` example with a few changes, in
[firmware/CameraWebServer](../firmware/CameraWebServer). The camera is already attached to the
board, but the firmware is still needed: it starts the camera, creates the Wi-Fi hotspot and serves
the images.

## Flashing the firmware

1. Open `firmware/CameraWebServer/CameraWebServer.ino` in the Arduino IDE (the folder name must stay
   `CameraWebServer`).
2. Install the **esp32** boards package (tested with 3.3.12).
3. Tools menu:

   | Setting | Value |
   |---|---|
   | Board | **ESP32 Wrover Module** |
   | PSRAM | **Enabled** |
   | Partition Scheme | **Huge APP (3MB No OTA/1MB SPIFFS)** |
   | Port | the CH340 port |

4. Upload. Open the Serial Monitor at 115200 baud and press the board's reset button. You should see:

   ```
   Hotspot 'ChessCam' started (password 'none')
   Camera Ready! Use 'http://192.168.4.1' to connect
   [alive] uptime 5s, clients 0, free heap ..., free psram 4...
   ```

   `free psram` must be well above 0. If it's 0, PSRAM isn't enabled in the Tools menu, and the
   board runs out of memory and stops answering on Wi-Fi.

The Arduino IDE on this PC also contains a command-line uploader
(`C:\Users\Jay\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe`) if you'd rather
flash from a terminal.

### What was changed from the stock example

| Change | Why |
|---|---|
| Pins chosen automatically from the board type (`board_config.h`) | `ESP32 Wrover Module` → Freenove WROVER pins; `ESP32S3 Dev Module` → Freenove S3 pins. The stock file selected a different board (ESP-EYE), so the camera failed to start |
| Wi-Fi hotspot `ChessCam` (open, no password) instead of joining a network | eduroam can't be used (see below). A password was rejected by Windows' PIN prompt, so the hotspot is open |
| Camera clock 10 MHz instead of 20 MHz | Less data for the ESP32 to move; aimed at the slow Wi-Fi (see [Camera speed](#camera-speed)) |
| Debug output off, a heartbeat line every 5 s | Shows whether the board is alive and whether PSRAM is found |
| Warning printed when PSRAM is missing | The most common cause of "connected but nothing loads" |

Image orientation is **not** set in the firmware. The PC sets it at startup over `/control`
(`camera.vflip` in `config.yaml`).

## Network

The ESP32 runs as a **hotspot**: network `ChessCam`, no password, ESP32 always at **192.168.4.1**.
The PC gets an address like 192.168.4.2.

- **eduroam doesn't work**: it needs a username + password login (WPA2-Enterprise), and campus
  networks block devices from reaching each other.
- While the PC's Wi-Fi is on `ChessCam` it has no internet over Wi-Fi. A wired Ethernet connection
  keeps working at the same time.
- Windows sometimes rejoins another network after the ESP32 restarts. If the camera stops, check
  that Wi-Fi is still on `ChessCam`.

### Camera endpoints

| URL | What |
|---|---|
| `http://192.168.4.1/` | The camera's own web page (settings, Start Stream, Save) |
| `http://192.168.4.1/capture` | One JPEG. This is what the program polls by default |
| `http://192.168.4.1:81/stream` | MJPEG stream. **Only one viewer at a time.** A browser tab on it blocks the program |
| `http://192.168.4.1/control?var=<name>&val=<n>` | Change a setting (`framesize`, `quality`, `vflip`, `hmirror`, `brightness`, ...) |
| `http://192.168.4.1/status` | All current settings as JSON |

`framesize` numbers depend on the esp32-camera version. With arduino-esp32 3.3.x: QVGA 320×240 = 6,
VGA 640×480 = 10, **SVGA 800×600 = 11** (used), XGA 1024×768 = 12. The program sets it by name and
checks the size of the image it gets back.

`firmware/CameraWebServer/snapshot.py` is a small standalone tool that saves snapshots:
`.\.venv\Scripts\python firmware\CameraWebServer\snapshot.py 192.168.4.1 -n 5 -d 1`.

## Mounting and orientation

- Mount the camera **above the board looking down**. Near-overhead is best: at a low side angle,
  tall pieces cover the squares behind them, which makes some moves ambiguous.
- The whole board, including the outer edge of the squares, must be in view.
- This camera delivers a **vertically mirrored** image, so `camera.vflip: true`. In a mirrored
  image the king and queen swap sides, so this matters. Check: printed board coordinates should read
  normally in the Camera window.
- Which side white sits on doesn't matter; it's detected from the starting position.
- Even, steady light helps. Avoid light that flickers or moves (people walking past a lamp).

## Camera speed

Measured on this setup: **0.4–2.5 frames per second**, with occasional stalls of 5–12 s and
timeouts.

What was ruled out:

| Check | Result |
|---|---|
| PC Wi-Fi link | 120–135 Mbps, 72–81% signal |
| PC Wi-Fi power saving | Off (Maximum Performance) |
| Lower resolution | QVGA was no faster than SVGA |

What points at the ESP32 itself:

- Pings to the ESP32 average 44–320 ms and reach 1 s; a device a few feet away should answer in 2–5 ms.
- A burst of small packets made it stop accepting connections.
- Throughput was 20–40 KB/s.

Likely causes (not yet confirmed):

- **Memory bus contention.** With PSRAM on, the stock firmware writes camera frames into PSRAM
  non-stop. On the classic ESP32, PSRAM shares a bus with the flash the Wi-Fi code runs from.
  The 10 MHz clock in the current firmware halves that load.
- **Power.** A weak USB supply can make the ESP32's Wi-Fi unreliable. Use a good USB port or a 5 V 2 A
  adapter.
- **Single-viewer stream.** Failed or extra connections (including a browser tab on the camera page)
  can hold the stream slot. The program uses `/capture` by default for this reason.

Effect on the program: a slow camera only delays when a move is read (the board must be seen still
in 2 consecutive frames). It doesn't cause wrong moves.
