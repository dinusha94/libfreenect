#!/usr/bin/env python3
"""Record a short RGB, depth or infrared clip from a Kinect (v1).

Uses libfreenect_sync through ctypes (no need to build the Cython wrapper)
and pipes frames to ffmpeg, so only numpy and the ffmpeg binary are needed.

Writes into the output directory:
  rgb.mp4        colour video (H.264)
  ir.mp4         infrared "night vision" video (with -s ir), 640x488
  depth.mp4      depth preview: near = bright, far = dark, no reading = black
  depth_raw.mkv  lossless 16-bit depth in millimetres (FFV1, gray16le)

Usage:
  python3 record_clip.py                 # 5 s of RGB+depth into ./clip
  python3 record_clip.py -d 10 -o clip2  # 10 seconds into ./clip2
  python3 record_clip.py -s rgb          # only RGB (-s depth: only depth)
  python3 record_clip.py -s ir           # infrared camera, works in the dark
  python3 record_clip.py -s ir --ir-gain 2  # brighten a dim IR image
  python3 record_clip.py --registered    # depth aligned to the RGB image
  Ctrl+C stops early (files are still saved).

Some USB controllers (e.g. Raspberry Pi 4) can't carry both camera streams
at once: frames stop arriving. The script then gives up after a few seconds;
record with -s rgb and -s depth separately instead.

IR comes from the camera on the right, lit by the Kinect's own dot projector,
so it sees in darkness (to ~4 m) but the image is speckled with dots. It
shares the video channel with RGB, so -s ir can't be combined with RGB.

Read the raw depth back into numpy (shape: frames x 480 x 640, uint16 mm):
  raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', 'clip/depth_raw.mkv',
                        '-f', 'rawvideo', '-pix_fmt', 'gray16le', '-'],
                       capture_output=True, check=True).stdout
  depth = np.frombuffer(raw, np.uint16).reshape(-1, 480, 640)
"""
import argparse
import ctypes
import ctypes.util
import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np

WIDTH, HEIGHT, FPS = 640, 480, 30
IR_HEIGHT = 488  # the IR sensor gives 8 more rows than the RGB one
STALL_TIMEOUT = 5.0  # seconds without a frame before giving up
FREENECT_VIDEO_RGB = 0
FREENECT_VIDEO_IR_8BIT = 2
FREENECT_DEPTH_REGISTERED = 4
FREENECT_DEPTH_MM = 5


def load_libfreenect_sync():
    candidates = [ctypes.util.find_library('freenect_sync'),
                  '/usr/local/lib/libfreenect_sync.so',
                  'libfreenect_sync.so.0', 'libfreenect_sync.so']
    for name in candidates:
        if not name:
            continue
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        sys.exit('Could not load libfreenect_sync shared library')

    getter_args = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32),
                   ctypes.c_int, ctypes.c_int]
    lib.freenect_sync_get_video.argtypes = getter_args
    lib.freenect_sync_get_depth.argtypes = getter_args
    lib.freenect_sync_stop.argtypes = []
    return lib


def grab(getter, index, fmt, dtype, shape):
    ptr = ctypes.c_void_p()
    ts = ctypes.c_uint32()
    if getter(ctypes.byref(ptr), ctypes.byref(ts), index, fmt) != 0:
        return None
    buf = (ctypes.c_char * (int(np.prod(shape)) * np.dtype(dtype).itemsize)).from_address(ptr.value)
    return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()


def ffmpeg_writer(path, in_pix_fmt, out_args, height=HEIGHT):
    cmd = ['ffmpeg', '-v', 'error', '-y',
           '-f', 'rawvideo', '-pix_fmt', in_pix_fmt,
           '-s', '%dx%d' % (WIDTH, height), '-r', str(FPS), '-i', '-'] + out_args + [path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def depth_preview(depth_mm, max_mm):
    """Map millimetres to 8-bit: near = bright, far = dark, 0 (no data) = black."""
    d = np.clip(depth_mm, 0, max_mm).astype(np.float32)
    img = 255 - (d * (255.0 / max_mm))
    img[depth_mm == 0] = 0
    return img.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-d', '--duration', type=float, default=5.0,
                        help='seconds to record (default: 5)')
    parser.add_argument('-o', '--outdir', default='clip',
                        help='output directory (default: clip)')
    parser.add_argument('-i', '--index', type=int, default=0,
                        help='Kinect device index (default: 0)')
    parser.add_argument('-s', '--streams', choices=('both', 'rgb', 'depth', 'ir'),
                        default='both',
                        help='what to record: both = RGB + depth (default)')
    parser.add_argument('--registered', action='store_true',
                        help='align depth to the RGB camera (DEPTH_REGISTERED)')
    parser.add_argument('--max-mm', type=int, default=4000,
                        help='farthest distance shown in depth.mp4 (default: 4000)')
    parser.add_argument('--ir-gain', type=float, default=1.0,
                        help='brightness multiplier for ir.mp4 (default: 1)')
    args = parser.parse_args()

    lib = load_libfreenect_sync()
    depth_fmt = FREENECT_DEPTH_REGISTERED if args.registered else FREENECT_DEPTH_MM
    want_rgb = args.streams in ('both', 'rgb')
    want_depth = args.streams in ('both', 'depth')
    want_ir = args.streams == 'ir'

    os.makedirs(args.outdir, exist_ok=True)
    h264 = ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p']
    writers = []
    if want_rgb:
        rgb_out = ffmpeg_writer(os.path.join(args.outdir, 'rgb.mp4'), 'rgb24', h264)
        writers.append(rgb_out)
    if want_ir:
        ir_out = ffmpeg_writer(os.path.join(args.outdir, 'ir.mp4'), 'gray', h264, IR_HEIGHT)
        writers.append(ir_out)
    if want_depth:
        prev_out = ffmpeg_writer(os.path.join(args.outdir, 'depth.mp4'), 'gray', h264)
        raw_out = ffmpeg_writer(os.path.join(args.outdir, 'depth_raw.mkv'), 'gray16le',
                                ['-c:v', 'ffv1', '-pix_fmt', 'gray16le'])
        writers += [prev_out, raw_out]

    target = int(args.duration * FPS)
    state = {'stop': False, 'frames': 0, 'last_frame': time.time(), 'error': None,
             'first_frame': None}

    def on_sigint(_sig, _frame):
        state['stop'] = True
    signal.signal(signal.SIGINT, on_sigint)

    # libfreenect_sync blocks forever if frames stop arriving, so capture runs
    # in a thread and the main thread watches for stalls.
    def capture():
        while state['frames'] < target and not state['stop']:
            if want_rgb:
                rgb = grab(lib.freenect_sync_get_video, args.index, FREENECT_VIDEO_RGB,
                           np.uint8, (HEIGHT, WIDTH, 3))
            if want_ir:
                ir = grab(lib.freenect_sync_get_video, args.index, FREENECT_VIDEO_IR_8BIT,
                          np.uint8, (IR_HEIGHT, WIDTH))
            if want_depth:
                depth = grab(lib.freenect_sync_get_depth, args.index, depth_fmt,
                             np.uint16, (HEIGHT, WIDTH))
            if ((want_rgb and rgb is None) or (want_depth and depth is None)
                    or (want_ir and ir is None)):
                state['error'] = 'Could not get frames from Kinect %d' % args.index
                break
            if state['stop']:
                break
            if want_rgb:
                rgb_out.stdin.write(rgb.tobytes())
            if want_ir:
                if args.ir_gain != 1.0:
                    ir = np.clip(ir * args.ir_gain, 0, 255).astype(np.uint8)
                ir_out.stdin.write(ir.tobytes())
            if want_depth:
                prev_out.stdin.write(depth_preview(depth, args.max_mm).tobytes())
                raw_out.stdin.write(depth.astype('<u2').tobytes())
            state['frames'] += 1
            state['last_frame'] = time.time()
            if state['first_frame'] is None:
                state['first_frame'] = state['last_frame']
        state['stop'] = True

    print('Recording %.1f s of %s from Kinect %d... (Ctrl+C to stop early)'
          % (args.duration, args.streams, args.index))
    thread = threading.Thread(target=capture, daemon=True)
    thread.start()
    stalled = False
    while thread.is_alive():
        thread.join(0.2)
        if thread.is_alive() and time.time() - state['last_frame'] > STALL_TIMEOUT:
            stalled = True
            state['stop'] = True
            break
    # Measured from the first frame, so camera start-up time isn't counted.
    elapsed = state['last_frame'] - state['first_frame'] if state['first_frame'] else 0

    if not stalled:
        lib.freenect_sync_stop()
    for proc in writers:
        proc.stdin.close()
        proc.wait()

    frames = state['frames']
    if frames == 0:
        for name in ('rgb.mp4', 'ir.mp4', 'depth.mp4', 'depth_raw.mkv'):
            path = os.path.join(args.outdir, name)
            if os.path.exists(path):
                os.remove(path)
    if state['error']:
        print(state['error'])
    if stalled:
        print('No frames for %.0f s, giving up.' % STALL_TIMEOUT)
        if args.streams == 'both':
            print('Your USB controller probably can\'t stream RGB and depth together;'
                  ' try -s rgb and -s depth separately.')
    if frames:
        print('Saved %d frames (%.1f s captured, %.1f fps) to %s/'
              % (frames, elapsed, (frames - 1) / elapsed if elapsed else 0, args.outdir))
    if stalled:
        # The capture thread is stuck inside libfreenect; skip normal teardown.
        sys.stdout.flush()
        os._exit(1)


if __name__ == '__main__':
    main()
