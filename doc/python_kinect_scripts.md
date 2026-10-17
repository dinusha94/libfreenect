# Kinect v1 Python scripts (Raspberry Pi 4)

Notes on the Python scripts in [`wrappers/python/`](../wrappers/python/) for
recording audio and video from a Kinect v1 and for the depth-based air mouse,
plus what we found out about the hardware along the way.

| Script | Purpose |
|---|---|
| [`record_mics.py`](../wrappers/python/record_mics.py) | Record the 4-mic array to `.wav` files and print per-mic levels |
| [`record_clip.py`](../wrappers/python/record_clip.py) | Record a short RGB, depth or infrared (night vision) clip |
| [`demo_air_mouse.py`](../wrappers/python/demo_air_mouse.py) | Control the mouse by pointing; push forward to click. `--dry-run` prints status over SSH |

Tested on a Raspberry Pi 4 Model B (8 GB), Python 3.13, libfreenect 0.7.5
installed in `/usr/local`, with no monitor attached (run over SSH).

---

## Setup

### Requirements

- libfreenect built and installed (`/usr/local/lib/libfreenect.so`,
  `libfreenect_sync.so`)
- `numpy`
- `ffmpeg` (for `record_clip.py`)
- The audio firmware `audios.bin` (for `record_mics.py`), installed in
  `/usr/local/share/libfreenect/`
- USB access to the Kinect without root (udev rules from the repo's
  `platform/linux/udev/`)

### Python `freenect` module

`record_mics.py` and `record_clip.py` **do not** need the Cython `freenect`
module: they load the libfreenect shared libraries directly with `ctypes`.

`demo_air_mouse.py` uses `import freenect`. The module is built by CMake
(`BUILD_PYTHON3=ON`) but installed to `/usr/local/lib/python3/dist-packages`,
which Python 3.13 does not search. Fix it once with:

```bash
echo /usr/local/lib/python3/dist-packages | sudo tee /usr/local/lib/python3.13/dist-packages/freenect.pth
```

or per shell session:

```bash
export PYTHONPATH=/usr/local/lib/python3/dist-packages
```

### Output location

All scripts write relative to the **current directory** unless you pass
`-o`. For example, running from `~/libfreenect/wrappers/python` with the
default settings creates `~/libfreenect/wrappers/python/mic_test/`.

---

## `record_mics.py`: microphone array test

Records all four microphones and the Kinect's noise-cancelled channel, saves
them as WAV files, and prints peak/RMS levels so a dead or noisy mic stands
out.

The Cython wrapper has no audio support, so this script uses libfreenect's
audio API (`freenect_start_audio`, audio-in callback) through `ctypes`, the
same flow as `examples/wavrecord.c`.

### Usage

```bash
cd ~/libfreenect/wrappers/python
python3 record_mics.py                 # 5 s into ./mic_test
python3 record_mics.py -d 10 -o test1  # 10 s into ./test1
```

| Option | Default | Meaning |
|---|---|---|
| `-d`, `--duration` | `5` | Seconds to record |
| `-o`, `--outdir` | `mic_test` | Output directory |
| `-i`, `--index` | `0` | Kinect device index |

Ctrl+C stops early and still saves the files.

### Output (16 kHz)

| File | Content |
|---|---|
| `mic1.wav` … `mic4.wav` | One file per mic, left to right, mono, 32-bit PCM |
| `all_mics.wav` | All four mics as one 4-channel file (compare them side by side in Audacity) |
| `cancelled.wav` | Noise-cancelled mix from the Kinect's audio DSP, 16-bit PCM |

Example level report:

```
channel     peak dBFS   RMS dBFS
mic1            -24.5      -42.3
mic2            -18.6      -39.0
mic3            -36.3      -57.4
mic4            -49.8      -62.4
cancelled       -30.3      -59.0
```

### Testing the mics

Ambient noise alone doesn't show much. Record ~10 s while tapping or speaking
near each mic in turn from left to right; each channel should get loud in its
turn. A channel that stays flat at a very low level is suspect.

libfreenect may print `audio: received an iso IN packet of strange length`.
This is harmless; the recording still works.

---

## `record_clip.py`: RGB / depth / infrared clips

Records a short clip using `libfreenect_sync` through `ctypes` and pipes
frames into `ffmpeg`.

### Usage

```bash
cd ~/libfreenect/wrappers/python
python3 record_clip.py -s rgb   -d 5 -o clip_rgb     # colour only
python3 record_clip.py -s depth -d 5 -o clip_depth   # depth only
python3 record_clip.py -s ir    -d 10 -o night       # infrared / night vision
python3 record_clip.py -s ir --ir-gain 2 -o night    # brighter IR
python3 record_clip.py -s depth --registered         # depth aligned to RGB
python3 record_clip.py                               # RGB + depth together (see limitation below)
```

| Option | Default | Meaning |
|---|---|---|
| `-d`, `--duration` | `5` | Seconds to record |
| `-o`, `--outdir` | `clip` | Output directory |
| `-i`, `--index` | `0` | Kinect device index |
| `-s`, `--streams` | `both` | `both` (RGB + depth), `rgb`, `depth` or `ir` |
| `--registered` | off | Use `DEPTH_REGISTERED` (depth aligned to the RGB image) instead of `DEPTH_MM` |
| `--max-mm` | `4000` | Farthest distance shown in the depth preview |
| `--ir-gain` | `1` | Brightness multiplier for `ir.mp4` |

Ctrl+C stops early and still saves the files.

### Output (30 fps)

| File | Content |
|---|---|
| `rgb.mp4` | Colour video, 640×480, H.264 |
| `ir.mp4` | Infrared video, 640×488, greyscale H.264 |
| `depth.mp4` | Depth preview: near = bright, far = dark, no reading = black |
| `depth_raw.mkv` | Lossless 16-bit depth in millimetres (FFV1, `gray16le`) |

Load the raw depth back into numpy (shape `frames × 480 × 640`, `uint16` mm):

```python
import subprocess, numpy as np
raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', 'clip/depth_raw.mkv',
                      '-f', 'rawvideo', '-pix_fmt', 'gray16le', '-'],
                     capture_output=True, check=True).stdout
depth = np.frombuffer(raw, np.uint16).reshape(-1, 480, 640)
```

### Limitation: RGB + depth together on the Pi 4

On this Pi 4 the USB controller can't carry the RGB and depth streams at the
same time: all frames are lost. The repo's own `freenect-camtest` fails the
same way, so this is the hardware, not the script. Changing USB ports doesn't
help, because the Kinect is a USB 2 device and all USB 2 ports on a Pi 4 share
one connection.

When frames stop arriving, the script gives up after 5 seconds with a message
instead of hanging, and removes the empty output files. Record `-s rgb` and
`-s depth` one after the other instead. A normal PC should handle both at once.

### Night vision (IR)

The Kinect's three front openings are, left to right:

1. **IR projector**, which shines an invisible dot pattern
2. **RGB camera**
3. **IR camera**, which sees the dots and is used to compute depth

`-s ir` records the IR camera. Because the projector supplies its own light,
it sees in complete darkness up to about 4–5 m, but the image is speckled with
dots. IR and RGB share the same video channel, so they can't be recorded at the
same time.

---

## `demo_air_mouse.py`: depth-based air mouse

Treats the nearest object held out in front of your body as a hand. Its
position moves the cursor, and a quick push towards the sensor (100 mm by
default) sends a left click. The cursor is frozen during the push so the
click lands where you aimed. It needs only the depth stream, so it works in
the dark.

Mount the Kinect facing the user, stand 1–2 m away and hold one hand about
20 cm in front of your body.

### Headless testing (`--dry-run`)

```bash
cd ~/libfreenect/wrappers/python
python3 demo_air_mouse.py --dry-run
```

With `--dry-run` the mouse isn't touched (no sudo or `/dev/uinput` needed).
Instead, a status line is printed, updated in place on a terminal or as one
line per update when output is redirected. Events print above it:

```
hand engaged at 908 mm
hand  908 mm | cursor 1024, 493 | push   0/100 mm | ready    | 29.9 fps
CLICK at 1035, 539
hand  635 mm | cursor  983, 530 | push 273/100 mm | released | 30.2 fps
no hand | nearest object  412 mm (range 600-2500 mm) | 30.0 fps
```

| Field | Meaning |
|---|---|
| `hand … mm` | Distance from the Kinect to the hand |
| `cursor x,y` | Where the mouse would be on the screen (`--width`/`--height`) |
| `push n/100 mm` | Forward travel within the push window; a click fires past `--push-mm` |
| `ready` / `released` | After a click it stays `released` until you pull the hand back |
| `no hand \| nearest object` | Nothing qualifies as a hand; shows the closest thing seen, to help when you're too close or too far |

### Real mouse control

```bash
sudo python3 demo_air_mouse.py --width 1920 --height 1080
```

This needs `python3-evdev` (`sudo apt install python3-evdev`) and write access
to `/dev/uinput`. To run it without sudo:

```bash
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/99-uinput.rules
sudo usermod -aG input $USER   # then reboot
```

`--preview` opens a debug window with OpenCV, so it needs a display.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--width`, `--height` | `1920`, `1080` | Screen size in pixels |
| `--box-w`, `--box-h` | `500`, `300` | Size of the hand movement area mapped to the full screen, mm |
| `--min-dist`, `--max-dist` | `600`, `2500` | Ignore anything closer / farther, mm |
| `--hand-depth` | `100` | Hand thickness measured from its nearest point, mm |
| `--reach` | `200` | How far the hand must be in front of the body, mm |
| `--min-hand-px` | `60` | Minimum hand blob size at 1 m (downsampled pixels) |
| `--push-mm` | `100` | Forward travel that counts as a click, mm |
| `--push-time` | `0.35` | Time window for the push, s |
| `--no-mirror` | off | Use if the cursor moves opposite to your hand |
| `--dry-run` | off | Print status and clicks instead of moving the mouse |
| `--status-hz` | `5` | Dry-run status updates per second |
| `--preview` | off | Debug window (needs `python3-opencv` and a display) |

---

## Hardware notes

### What libfreenect offers

**Done by the Kinect's own chips**
- Depth computation from the IR dot pattern (PrimeSense chip). Converting
  raw values to millimetres (`DEPTH_MM`) is done by libfreenect on the host.
- Camera settings via `freenect_set_flag`: auto exposure, auto white
  balance, anti-flicker, `RAW_COLOR`, `MIRROR_DEPTH` / `MIRROR_VIDEO`,
  `NEAR_MODE` (Kinect for Windows only)
- Manual exposure (`freenect_set_exposure`) and IR projector brightness
  (`freenect_set_ir_brightness`)
- YUV video modes (converted in the camera, 15 fps only)
- Noise-cancelled audio channel (the `cancelled.wav` stream)

**Motor and sensors**
- Tilt motor, about ±27° (`freenect_set_tilt_degs`, `freenect_get_tilt_status`)
- Accelerometer (`freenect_get_mks_accel`), useful for levelling a point cloud
- Front LED (`freenect_set_led`): off, green, red, yellow, blinking

**Done in software by libfreenect**
- Converting the raw colour (Bayer) image to RGB
- Aligning depth to the RGB image (`DEPTH_REGISTERED`, `freenect_map_rgb_to_depth`)
- Converting depth pixels to world coordinates (`freenect_camera_to_world`)
- Unpacking the packed depth/IR formats

**Formats**
- Colour: 1280×1024 at ~10 fps or 640×480 at 30 fps, RGB or Bayer
- IR: 1280×1024 at ~10 fps or 640×488 at 30 fps, 8-bit or 10-bit
- Depth: 640×480 at 30 fps, raw 11/10-bit or millimetres

**Other**: audio firmware upload, opening a device by serial number, and
`fakenect` to record and replay a session without the Kinect.

### Point clouds

Each depth pixel in millimetres plus its image position gives a real
(X, Y, Z) point; `freenect_camera_to_world()` does this with the calibration
stored in the Kinect. On the Pi, a depth-only point cloud works. A coloured
point cloud of a static scene is possible by grabbing an RGB frame and then a
`DEPTH_REGISTERED` frame one after the other. The example programs
`freenect-glpclview` and `freenect-cpp_pcview` show a live point cloud but need
a display.

### Gesture and face recognition

The Kinect v1 can't run models: its chips are fixed-function, and skeleton
tracking on the Xbox ran on the console. Any recognition runs on the Pi.

- **Gestures from depth alone** (as in `demo_air_mouse.py`): full 30 fps,
  works in the dark, but no hand shapes.
- **Hand landmarks**: MediaPipe Hands on the RGB image (~10–15 fps on a Pi 4).
  It may not have a build for Python 3.13 on ARM yet; use a Python 3.11 venv
  if needed.
- **Faces**: OpenCV YuNet (`cv2.FaceDetectorYN`) for detection, SFace
  (`cv2.FaceRecognizerSF`) for recognition. Detection often works on the IR
  image too, giving face detection in the dark.

The speeds above are typical figures, not measured on this Pi.

### Kernel webcam driver

When the Kinect is plugged in, the kernel may load `gspca_kinect` for the
camera. libfreenect detaches it automatically, so it doesn't block the
scripts. To stop it from loading:

```bash
echo "blacklist gspca_kinect" | sudo tee /etc/modprobe.d/blacklist-kinect.conf
```
