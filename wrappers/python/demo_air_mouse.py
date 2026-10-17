#!/usr/bin/env python3
"""Kinect "air mouse": point at the screen with your hand, push forward to click.

Mount the Kinect on top of the screen, facing the user. Stand 1-2 m away and
raise a hand in front of your body. The closest object to the sensor is taken
as the hand; its position drives the cursor and a quick push toward the screen
sends a left click.

Clicks are injected through a virtual uinput pointer, so they work on X11 and
Wayland (Raspberry Pi OS Bookworm). This needs python3-evdev and write access
to /dev/uinput (run with sudo, or see the udev rule printed by --help).

    python3 demo_air_mouse.py --width 1920 --height 1080 --preview
    python3 demo_air_mouse.py --dry-run          # print events, no mouse
"""
import argparse
import math
import sys
import time
from collections import deque

import numpy as np

import freenect

# Kinect v1 depth camera intrinsics at 640x480 (approximate)
FOCAL_PX = 580.0
CX, CY = 320.0, 240.0


class OneEuroFilter:
    """Low-latency smoothing: steady when still, responsive when moving fast."""

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, x, t):
        if self.t is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


class Pointer:
    """Absolute pointer + left button via uinput, or prints in dry-run mode."""

    def __init__(self, width, height, dry_run):
        self.dry_run = dry_run
        if dry_run:
            return
        try:
            from evdev import UInput, AbsInfo, ecodes as e
        except ImportError:
            sys.exit("python3-evdev is missing: sudo apt install python3-evdev")
        self.e = e
        caps = {
            e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT],
            e.EV_ABS: [
                (e.ABS_X, AbsInfo(0, 0, width - 1, 0, 0, 0)),
                (e.ABS_Y, AbsInfo(0, 0, height - 1, 0, 0, 0)),
            ],
        }
        try:
            self.ui = UInput(caps, name="kinect-air-mouse")
        except (PermissionError, OSError) as err:
            sys.exit("Cannot open /dev/uinput (%s). Run with sudo or see --help." % err)

    def move(self, x, y):
        if self.dry_run:
            return
        self.ui.write(self.e.EV_ABS, self.e.ABS_X, int(x))
        self.ui.write(self.e.EV_ABS, self.e.ABS_Y, int(y))
        self.ui.syn()

    def click(self, x, y):
        if self.dry_run:
            print("CLICK at %4d,%4d" % (x, y))
            return
        self.move(x, y)
        self.ui.write(self.e.EV_KEY, self.e.BTN_LEFT, 1)
        self.ui.syn()
        time.sleep(0.03)
        self.ui.write(self.e.EV_KEY, self.e.BTN_LEFT, 0)
        self.ui.syn()

    def close(self):
        if not self.dry_run:
            self.ui.close()


def find_hand(depth, args):
    """Return (u, v, z_mm, mask) of the nearest blob, or None.

    depth is a downsampled DEPTH_MM frame (0 = no reading)."""
    valid = (depth > args.min_dist) & (depth < args.max_dist)
    if not valid.any():
        return None
    # A low percentile instead of min() ignores single noisy pixels
    near, body = np.percentile(depth[valid], [0.5, 25])
    # Only a hand held out in front of the body counts, not the body itself
    if body - near < args.reach:
        return None
    mask = valid & (depth < near + args.hand_depth)
    vs, us = np.nonzero(mask)
    # Blob area shrinks with distance squared; require a hand-sized blob
    min_px = args.min_hand_px * (1000.0 / near) ** 2
    if len(us) < min_px:
        return None
    z = float(np.median(depth[vs, us]))
    return float(us.mean()), float(vs.mean()), z, mask


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Run without sudo: echo \'KERNEL=="uinput", GROUP="input", MODE="0660"\' | '
               'sudo tee /etc/udev/rules.d/99-uinput.rules; sudo usermod -aG input $USER; reboot')
    p.add_argument("--width", type=int, default=1920, help="screen width in pixels")
    p.add_argument("--height", type=int, default=1080, help="screen height in pixels")
    p.add_argument("--box-w", type=float, default=500, help="interaction box width, mm")
    p.add_argument("--box-h", type=float, default=300, help="interaction box height, mm")
    p.add_argument("--min-dist", type=float, default=600, help="ignore anything closer, mm")
    p.add_argument("--max-dist", type=float, default=2500, help="ignore anything farther, mm")
    p.add_argument("--hand-depth", type=float, default=100, help="hand thickness from nearest point, mm")
    p.add_argument("--reach", type=float, default=200, help="hand must be this far in front of the body, mm")
    p.add_argument("--min-hand-px", type=float, default=60, help="min hand blob size at 1 m (downsampled px)")
    p.add_argument("--push-mm", type=float, default=100, help="forward travel that counts as a press, mm")
    p.add_argument("--push-time", type=float, default=0.35, help="time window for the push, s")
    p.add_argument("--no-mirror", action="store_true", help="use if the cursor moves opposite to your hand")
    p.add_argument("--dry-run", action="store_true", help="print clicks instead of moving the mouse")
    p.add_argument("--preview", action="store_true", help="show a debug window (needs python3-opencv)")
    args = p.parse_args()

    cv2 = None
    if args.preview:
        try:
            import cv2
        except ImportError:
            sys.exit("--preview needs python3-opencv: sudo apt install python3-opencv")

    pointer = Pointer(args.width, args.height, args.dry_run)
    fx, fy = OneEuroFilter(), OneEuroFilter()
    history = deque()          # (t, z, screen_x, screen_y)
    center = None              # hand position (mm) when it was raised
    armed = True
    rest_z = None
    lost_since = None
    sx = sy = 0.0
    step = 2                   # downsample 640x480 -> 320x240 for speed

    print("Raise a hand towards the screen. Push forward to click. Ctrl+C to quit.")
    try:
        while True:
            frame = freenect.sync_get_depth(format=freenect.DEPTH_MM)
            if frame is None:
                sys.exit("Kinect not found (check lsusb and the udev rules)")
            depth = frame[0][::step, ::step].astype(np.float32)
            t = time.time()

            hand = find_hand(depth, args)
            if hand is None:
                # Disengage after a short gap so the next raise re-centres the box
                if lost_since is None:
                    lost_since = t
                elif t - lost_since > 0.5 and center is not None:
                    center = None
                    fx.reset(); fy.reset(); history.clear()
                    armed = True
                    print("hand lost")
            else:
                lost_since = None
                u, v, z, mask = hand
                # Pixel -> metric coordinates at the hand's depth
                x_mm = (u * step - CX) * z / FOCAL_PX
                y_mm = (v * step - CY) * z / FOCAL_PX
                if center is None:
                    center = (x_mm, y_mm)
                    print("hand engaged at %.0f mm" % z)

                dx = (x_mm - center[0]) / args.box_w
                dy = (y_mm - center[1]) / args.box_h
                if not args.no_mirror:
                    dx = -dx
                tx = np.clip((0.5 + dx) * args.width, 0, args.width - 1)
                ty = np.clip((0.5 + dy) * args.height, 0, args.height - 1)

                while history and t - history[0][0] > args.push_time:
                    history.popleft()
                travel = history[0][1] - z if history else 0.0

                # Freeze the cursor while pushing so the click lands where you aimed
                if travel < args.push_mm * 0.4:
                    sx, sy = fx(tx, t), fy(ty, t)
                    pointer.move(sx, sy)
                history.append((t, z, sx, sy))

                if armed and travel > args.push_mm:
                    # Click where the cursor was before the push started
                    _, _, px, py = history[0]
                    pointer.click(px, py)
                    armed, rest_z = False, z
                elif not armed and z > rest_z + args.push_mm * 0.5:
                    armed = True

            if cv2 is not None:
                vis = np.clip(depth / args.max_dist * 255, 0, 255).astype(np.uint8)
                vis = cv2.cvtColor(255 - vis, cv2.COLOR_GRAY2BGR)
                if hand is not None:
                    vis[hand[3]] = (0, 200, 0) if armed else (0, 200, 255)
                    cv2.circle(vis, (int(hand[0]), int(hand[1])), 5, (0, 0, 255), -1)
                cv2.imshow("air mouse", cv2.resize(vis, (640, 480)))
                if cv2.waitKey(1) == 27:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        pointer.close()
        freenect.sync_stop()


if __name__ == "__main__":
    main()
