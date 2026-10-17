#!/usr/bin/env python3
"""Record the Kinect (v1) 4-mic array to .wav files for testing the mics.

The Cython wrapper doesn't expose audio, so this talks to libfreenect
directly through ctypes (same flow as examples/wavrecord.c).

Writes into the output directory:
  mic1.wav .. mic4.wav   one file per microphone (32-bit PCM, 16 kHz, mono)
  all_mics.wav           all four mics as a 4-channel file
  cancelled.wav          the Kinect's noise-cancelled stream (16-bit PCM)

Then prints per-mic peak/RMS levels so a dead or noisy mic stands out.

Usage:
  python3 record_mics.py              # record 5 seconds into ./mic_test
  python3 record_mics.py -d 10 -o out # record 10 seconds into ./out
  Ctrl+C stops early (files are still saved).

Needs the audio firmware (audios.bin) installed, e.g. in
/usr/local/share/libfreenect, and permission to access the USB device.
"""
import argparse
import ctypes
import ctypes.util
import os
import signal
import sys
import wave

import numpy as np

SAMPLE_RATE = 16000
FREENECT_DEVICE_AUDIO = 0x04
FREENECT_LOG_WARNING = 2

audio_in_cb_t = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int16), ctypes.c_void_p)


def load_libfreenect():
    candidates = [ctypes.util.find_library('freenect'),
                  '/usr/local/lib/libfreenect.so',
                  'libfreenect.so.0', 'libfreenect.so']
    for name in candidates:
        if not name:
            continue
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        sys.exit('Could not load libfreenect shared library')

    vp = ctypes.c_void_p
    lib.freenect_init.argtypes = [ctypes.POINTER(vp), vp]
    lib.freenect_shutdown.argtypes = [vp]
    lib.freenect_set_log_level.argtypes = [vp, ctypes.c_int]
    lib.freenect_select_subdevices.argtypes = [vp, ctypes.c_int]
    lib.freenect_num_devices.argtypes = [vp]
    lib.freenect_open_device.argtypes = [vp, ctypes.POINTER(vp), ctypes.c_int]
    lib.freenect_close_device.argtypes = [vp]
    lib.freenect_process_events.argtypes = [vp]
    lib.freenect_set_audio_in_callback.argtypes = [vp, audio_in_cb_t]
    lib.freenect_start_audio.argtypes = [vp]
    lib.freenect_stop_audio.argtypes = [vp]
    return lib


def write_wav(path, data, sampwidth):
    """data: int array shaped (samples,) or (samples, channels)."""
    channels = 1 if data.ndim == 1 else data.shape[1]
    with wave.open(path, 'wb') as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(np.ascontiguousarray(data).astype('<i%d' % sampwidth).tobytes())


def level_db(x, full_scale):
    x = x.astype(np.float64) / full_scale
    rms = np.sqrt(np.mean(x ** 2)) if x.size else 0.0
    peak = np.max(np.abs(x)) if x.size else 0.0
    to_db = lambda v: 20 * np.log10(v) if v > 0 else float('-inf')
    return to_db(peak), to_db(rms)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('-d', '--duration', type=float, default=5.0,
                        help='seconds to record (default: 5)')
    parser.add_argument('-o', '--outdir', default='mic_test',
                        help='output directory (default: mic_test)')
    parser.add_argument('-i', '--index', type=int, default=0,
                        help='Kinect device index (default: 0)')
    args = parser.parse_args()

    lib = load_libfreenect()
    ctx = ctypes.c_void_p()
    if lib.freenect_init(ctypes.byref(ctx), None) < 0:
        sys.exit('freenect_init() failed')
    lib.freenect_set_log_level(ctx, FREENECT_LOG_WARNING)
    lib.freenect_select_subdevices(ctx, FREENECT_DEVICE_AUDIO)

    if lib.freenect_num_devices(ctx) < 1:
        lib.freenect_shutdown(ctx)
        sys.exit('No Kinect found')

    dev = ctypes.c_void_p()
    if lib.freenect_open_device(ctx, ctypes.byref(dev), args.index) < 0:
        lib.freenect_shutdown(ctx)
        sys.exit('Could not open audio device (firmware/permissions?)')

    mics = [[], [], [], []]
    cancelled = []
    target = int(args.duration * SAMPLE_RATE)
    state = {'samples': 0, 'stop': False}

    def on_audio(_dev, n, m1, m2, m3, m4, canc, _unknown):
        for buf, ptr in zip(mics, (m1, m2, m3, m4)):
            buf.append(np.ctypeslib.as_array(ptr, shape=(n,)).copy())
        cancelled.append(np.ctypeslib.as_array(canc, shape=(n,)).copy())
        state['samples'] += n
        if state['samples'] >= target:
            state['stop'] = True

    callback = audio_in_cb_t(on_audio)  # keep a reference while streaming
    lib.freenect_set_audio_in_callback(dev, callback)

    def on_sigint(_sig, _frame):
        state['stop'] = True
    signal.signal(signal.SIGINT, on_sigint)

    print('Recording %.1f s from Kinect %d... (Ctrl+C to stop early)'
          % (args.duration, args.index))
    lib.freenect_start_audio(dev)
    while not state['stop'] and lib.freenect_process_events(ctx) >= 0:
        pass
    lib.freenect_stop_audio(dev)
    lib.freenect_close_device(dev)
    lib.freenect_shutdown(ctx)

    if state['samples'] == 0:
        sys.exit('No audio received')

    os.makedirs(args.outdir, exist_ok=True)
    mic_data = [np.concatenate(b) for b in mics]
    canc_data = np.concatenate(cancelled)

    for i, data in enumerate(mic_data, 1):
        write_wav(os.path.join(args.outdir, 'mic%d.wav' % i), data, 4)
    write_wav(os.path.join(args.outdir, 'all_mics.wav'), np.stack(mic_data, axis=1), 4)
    write_wav(os.path.join(args.outdir, 'cancelled.wav'), canc_data, 2)

    print('Saved %d samples (%.2f s) to %s/'
          % (state['samples'], state['samples'] / SAMPLE_RATE, args.outdir))
    print('\n%-10s %10s %10s' % ('channel', 'peak dBFS', 'RMS dBFS'))
    for i, data in enumerate(mic_data, 1):
        peak, rms = level_db(data, 2.0 ** 31)
        print('%-10s %10.1f %10.1f' % ('mic%d' % i, peak, rms))
    peak, rms = level_db(canc_data, 2.0 ** 15)
    print('%-10s %10.1f %10.1f' % ('cancelled', peak, rms))


if __name__ == '__main__':
    main()
