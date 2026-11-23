# Driver Monitor (proof-of-concept)

Detects **drowsiness** (eyes closed, PERCLOS, yawning), **inattention** (head turned away / face lost)
and **intoxication** (breath-alcohol sensor) in real time. No training needed: it uses MediaPipe Face Mesh
(pretrained, bundled in the pip package) plus classic landmark geometry (EAR, MAR, head pose via solvePnP).

Files: `monitor.py` (main loop) | `state.py` (**all thresholds**) | `features.py` (EAR/MAR/head pose) |
`camera.py` (PiCamera / webcam / video) | `sensors.py` (alcohol) | `alerts.py` (buzzer) | `test_logic.py`

---------------------------------------------------------------------------------------------------
## 1. Run on your PC (Windows/Mac/Linux, Python 3.10-3.12)
Create the virtual environment:
```
py -3.12 -m venv venv
```
Use Python 3.10, 3.11, or 3.12 for this project. If `py -3.12` is unavailable,
install Python 3.12 first; `python` by itself may select a newer, unsupported version.

Activate it using the command for your shell:
- Windows PowerShell:
  ```
  .\venv\Scripts\Activate.ps1
  ```
  If PowerShell says script execution is disabled, allow scripts for this PowerShell
  session only, then activate:
  ```
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
  .\venv\Scripts\Activate.ps1
  ```
- Windows Command Prompt: `venv\Scripts\activate.bat`
- Linux/Mac: `source venv/bin/activate`

Then install dependencies and run:
```
    pip install -r requirements.txt
    python test_logic.py           # offline sanity check
    python monitor.py              # laptop webcam, profile "pc", simulated alcohol sensor
```

Keys: `q` quit | `c` recalibrate | `a` toggle simulated alcohol.
At start, look straight at the camera with eyes open for 3 s (personal calibration).
No webcam? `python monitor.py --source some_video.mp4`.

## 2. Deploy on the Raspberry Pi 3B with the CSI Pi Camera

### 2.1 Connect the camera (Pi powered OFF)
1. Find the **CAMERA** port: the flat ribbon connector between the HDMI port and the 3.5 mm audio jack.
2. Gently pull up the plastic locking tab (about 2 mm, do not force it off).
3. Insert the ribbon with the **silver contacts facing the HDMI port** (blue backing toward the USB/Ethernet
   side). Push it in straight and fully, then press the tab back down. The ribbon must not be twisted
   or creased.
4. At the camera board end, do the same: ribbon fully seated, tab locked, contacts against the connector.
   If unsure, check the Raspberry Pi camera documentation picture for your exact module.
5. Mount the camera at roughly eye level or slightly below, 50-70 cm from the face, facing you, with light
   on the face (not a bright window behind you). The face should fill about a third to half of the frame.
   If the image appears upside down, use `--rotate 180`.
6. Power on, then check it is detected:
   ```
   rpicam-hello --list-cameras                     # (older OS: libcamera-hello)
   rpicam-still -o test.jpg                        # takes a photo, works without a screen
   ```
   Nothing listed? Power off and reseat both ends of the ribbon (the most common cause). Also make sure
   the legacy camera option is OFF in `sudo raspi-config` and that `/boot/firmware/config.txt` contains
   `camera_auto_detect=1`.

### 2.2 Install and run
Use **64-bit** Raspberry Pi OS (Bookworm). The mediapipe wheels do not exist for the 32-bit OS.
```
sudo apt update
sudo apt install -y git python3-venv python3-picamera2 python3-gpiozero python3-lgpio python3-spidev
git clone <your-repo-url> && cd driver-monitor
python3 -m venv venv --system-site-packages     # lets the venv see the apt packages above
source venv/bin/activate
pip install -r requirements-pi.txt
python monitor.py --profile pi --picamera --width 320 --sensor mq3 --buzzer 17
```
Useful flags: `--headless` (SSH, no display) | `--rotate 180` | `--sensor sim` (no MQ-3 connected) |
`--log run.csv` (record every frame's numbers, handy as evidence for your report).
Check the camera from Python first if something fails:
`python -c "from picamera2 import Picamera2; print(Picamera2.global_camera_info())"`

### 2.3 Buzzer and alcohol sensor wiring (GPIO)
Power the Pi with a proper 5 V / 2.5 A supply. Pin numbers below are the **physical header pins**.

| Part | Connection |
|---|---|
| Active buzzer (+) | GPIO17 = pin 11 (through an NPN transistor, see note) |
| Active buzzer (-) | GND = pin 6 |
| MCP3008 pin 16 VDD and pin 15 VREF | 3.3 V = pin 1 |
| MCP3008 pin 14 AGND and pin 9 DGND | GND = pin 9 |
| MCP3008 pin 13 CLK | GPIO11 SCLK = pin 23 |
| MCP3008 pin 12 DOUT | GPIO9 MISO = pin 21 |
| MCP3008 pin 11 DIN | GPIO10 MOSI = pin 19 |
| MCP3008 pin 10 CS | GPIO8 CE0 = pin 24 |
| MCP3008 pin 1 CH0 | MQ-3 analog output **through the divider below** |
| MQ-3 VCC | 5 V = pin 2 |
| MQ-3 GND | GND = pin 6 or 9 |

**Voltage divider (mandatory):** MQ-3 AO -> 10 kΩ -> node -> 20 kΩ -> GND, and connect the node to MCP3008 CH0.
The MQ-3 module outputs up to 5 V and the Pi/ADC only tolerate 3.3 V.
**Buzzer note:** a GPIO pin can only supply a few mA. If the buzzer is weak or the Pi misbehaves, drive it
with an NPN transistor (e.g. 2N2222: GPIO17 -> 1 kΩ -> base, emitter -> GND, buzzer between 5 V and collector).
**Enable SPI:** `sudo raspi-config` -> Interface Options -> SPI -> Enable, then reboot.
**Warm-up:** let the MQ-3 heat for several minutes (longer is better) and start the program in clean air,
because it records the clean-air baseline at start-up.
**Safe demo:** hold a cotton ball with a little hand sanitizer near the sensor. Do not drink alcohol for it.

---------------------------------------------------------------------------------------------------
## 3. Thresholds you will adjust (all in `state.py`, each one is commented there)

Workflow: run, deliberately do each behaviour, watch the numbers drawn on the video (EAR, MAR, yaw, pitch),
change the value, rerun. Per-device values live in `PROFILES` in `state.py`: `--profile pc` or `--profile pi`.

| Setting | What it controls | Change it when | PC start | Pi start |
|---|---|---|---|---|
| `ear_ratio` | how closed counts as "eyes closed" (x your own open-eye EAR) | false closed-eye alerts: lower (0.65). Misses closures: raise (0.80) | 0.72 | 0.70 (320 px landmarks are noisier) |
| `microsleep_s` | seconds of continuous closure for DANGER | demo feels slow: 1.0. Blinks trigger it: raise | 1.5 | 1.5 (not below 1.0 at low FPS) |
| `perclos_window_s`, `perclos_limit` | share of time eyes closed over a window | want a quicker reaction in a short demo: window 15 | 30 s / 0.25 | same |
| `mar_open` | mouth opening that counts as a yawn | talking triggers it: raise. Yawns missed: lower. Camera below the face exaggerates MAR | 0.60 | check on screen, expect to raise it |
| `yawn_s`, `yawns_for_fatigue` | yawn duration, yawns needed for FATIGUE | demo: `yawns_for_fatigue = 2` | 1.2 s / 3 | same |
| `yaw_limit`, `pitch_limit` | degrees of head turn / tilt from straight ahead | complains while looking at the road: widen. Misses glances: tighten | 25 / 20 | 30 / 25 (camera is off your line of sight) |
| `distract_s` | seconds looking away or face missing for DANGER | demo: 1.5 | 2.0 | 2.0 |
| MQ-3 `--alcohol-delta` (Pi only) | rise above clean-air baseline that counts as alcohol | false alarms: raise (0.12-0.2). Sanitizer ignored: lower (0.03-0.05) | n/a | 0.08 |

The Pi values are educated starting points, not measured ones. Tune them on the real setup.
The logic is time-based, so a lower frame rate on the Pi changes responsiveness slightly, not the timings.

---------------------------------------------------------------------------------------------------
## 4. Honest limitations (say these in your defense before the panel does)
- Intoxication is **not** detected from the camera; it comes from an uncalibrated MQ-3 sensor and only
  indicates alcohol in the air near the sensor, not blood alcohol level.
- Thresholds were tuned by hand, not validated on a dataset. EAR struggles with glasses, heavy glare and
  low light. A real product would use infrared cameras and validated data (e.g. NTHU-DDD, YawDD).
- Performance on a Pi 3B is limited (1 GB RAM, slow CPU), hence the low resolution.
