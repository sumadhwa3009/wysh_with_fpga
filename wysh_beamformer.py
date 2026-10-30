#!/usr/bin/env python3
"""
WYSH beamformer - Raspberry Pi side
FPGA (Tang Nano 9K) --UART 3 Mbaud--> Pi --> MVDR / delay-and-sum --> headphones

Stream format sent by the FPGA (one frame per audio sample):
    0xA5 | ch0_hi ch0_lo | ch1_hi ch1_lo | ... | ch7_hi ch7_lo     (17 bytes, int16 big-endian)
    ch0..ch5 = mics 0..5 (D0..D2),  ch6 = empty slot,  ch7 = centre mic (D3)

Usage:
    python3 wysh_beamformer.py                 # uses SERIAL_PORT below
    python3 wysh_beamformer.py /dev/ttyUSB0    # override the port

Needs only numpy + pyserial. Audio goes out through `aplay` (no PyAudio).

Controls while running:
    <number>   steer the beam, degrees  (0 = straight ahead, +90 = right, -90 = left, +-180 = behind)
    s          sweep -90..+90 deg, print output level per angle, beam goes to the loudest
    m          toggle MVDR <-> delay-and-sum
    g <x>      output gain, e.g.  g 5
    q          quit
"""
import os
import sys
import time
import math
import queue
import threading
import subprocess
import multiprocessing as mp

import numpy as np
import serial

# =============================== CONFIG ===============================
SERIAL_PORT = '/dev/serial0'      # FPGA UART on Pi pin 10 (GPIO15). USB device would be /dev/ttyUSB0 or /dev/ttyACM0
SERIAL_BAUD = 3000000

SAMPLE_RATE = 27e6 / 26 / 64      # ~16226 Hz, fixed by the FPGA clock divider
ALSA_DEVICE = 'plughw:Headphones' # Pi 3.5 mm jack (see `aplay -l`)
ALSA_BUFFER_US = 400000           # aplay -B, in microseconds (400 ms). Raise if underruns appear.

# --- which FPGA channels to use and where those mics physically sit -------------
# 6-mic ring: ch0..ch5 = mics 0..5.  Coordinates in metres: x = right, y = straight ahead.
# Azimuth 0 = straight ahead (+y), +90 = right (+x), -90 = left.
#
# !! Two things below are ASSUMPTIONS until you check them on your board:
#    RING_RADIUS_M    - measure centre-of-board to a ring mic (or half the distance between opposite mics)
#    RING_ANGLES_DEG  - angle of each channel around the ring, 0 = top of the board, clockwise.
#                       Default assumes ch0..ch5 run around the ring in order, 60 deg apart.
#                       Use the `s` sweep to check this against a sound source.
RING_RADIUS_M = 0.046
RING_ANGLES_DEG = [0, 60, 120, 180, 240, 300]

MIC_CHANNELS = [0, 1, 2, 3, 4, 5]
_ang = np.radians(RING_ANGLES_DEG)
MIC_POSITIONS = RING_RADIUS_M * np.stack([np.sin(_ang), np.cos(_ang)], axis=1)

# Add the centre mic (ch7) at the origin by using:
#   MIC_CHANNELS = [0, 1, 2, 3, 4, 5, 7]
#   MIC_POSITIONS = np.vstack([MIC_POSITIONS, [0.0, 0.0]])

AZIMUTH_SIGN = 1.0     # set to -1.0 if the beam turns out mirrored left <-> right

# --- DSP ----------------------------------------------------------------------
NFFT = 256             # 63 Hz bins, 16 ms window
HOP = NFFT // 2        # sqrt-Hann analysis+synthesis at 50 % overlap = perfect reconstruction
FREQ_MIN = 300.0       # MVDR is applied in this band; outside it plain delay-and-sum
FREQ_MAX = 4000.0
HIGH_PASS_HZ = 100.0   # bins below this are zeroed (removes DC and rumble)
ALPHA = 0.98           # covariance smoothing per frame (~127 frames/s -> ~0.4 s memory)
LOADING = 0.05         # diagonal loading, relative to signal power (higher = safer, closer to DAS)
WEIGHT_EVERY = 4       # MVDR weights are recomputed every N hops (covariance still updates every hop)
OUTPUT_GAIN = 7.0
QUEUE_CHUNKS = 200     # max serial chunks waiting for the DSP (~3 s of audio)

SPEED_OF_SOUND = 343.0

# --- stream constants (must match the FPGA) ------------------------------------
SYNC = 0xA5
SYNC_BYTE = b'\xa5'
N_STREAM_CH = 8
FRAME = 1 + 2 * N_STREAM_CH   # 17 bytes

# =============================== STATE ================================
running = True
azimuth = 0.0
mode = 'mvdr'
gain = OUTPUT_GAIN
stats = {'underrun': 0}
dsp_t = {'busy': 0.0, 'calls': 0, 'max': 0.0}          # time spent inside bf.process()
level = {'sum_sq': 0.0, 'n': 0, 'last_rms': 0.0}       # output level (before gain)
lock = threading.Lock()

# Shared counters between the reader process and the main process.
# Index: 0 = resync bytes discarded, 1 = frames received, 2 = frames dropped (DSP too slow)
CNT_RESYNC, CNT_FRAMES, CNT_DROPPED = 0, 1, 2


# ============================ FRAME PARSER ============================
def extract_frames(buf, cnt):
    """Pull complete, aligned frames out of bytearray `buf` (consumed in place).
    Locks only when 4 consecutive frames start with the sync byte.
    Returns int16 array (n, 8) or None."""
    need = 4 * FRAME
    slipped = 0
    while len(buf) >= need and not all(buf[i * FRAME] == SYNC for i in range(4)):
        del buf[0]
        slipped += 1
    if slipped:
        cnt[CNT_RESYNC] += slipped

    if len(buf) < need:
        return None

    n = len(buf) // FRAME
    raw = np.frombuffer(bytes(buf[:n * FRAME]), np.uint8).reshape(n, FRAME)
    bad = np.flatnonzero(raw[:, 0] != SYNC)
    if len(bad):
        n = int(bad[0])
        raw = raw[:n]
    if n == 0:
        return None
    pcm = np.ascontiguousarray(raw[:, 1:]).view('>i2').reshape(n, N_STREAM_CH)
    del buf[:n * FRAME]
    return pcm


# ============================= READER PROCESS =========================
def reader_proc(port, baud, out_q, cnt):
    """Runs in its own process so the DSP work can't take the GIL away from it.
    Does only: read bytes, parse frames, select channels, push to the queue."""
    try:
        os.sched_setaffinity(0, {3})   # keep the reader on core 3, away from the DSP
    except Exception:
        pass
    try:
        os.nice(-10)                   # higher priority; needs root
    except OSError:
        pass

    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"Cannot open {port}: {e}", flush=True)
        return

    buf = bytearray()
    while True:
        try:
            data = ser.read(4096)          # returns after 4096 bytes or 0.1 s
        except Exception as e:
            print(f"\nserial error: {e}", flush=True)
            time.sleep(0.5)
            continue
        if not data:
            continue
        buf += data
        pcm = extract_frames(buf, cnt)
        if pcm is None:
            continue
        sel = pcm[:, MIC_CHANNELS].astype(np.float64) / 32768.0
        cnt[CNT_FRAMES] += len(sel)
        try:
            out_q.put_nowait(sel)
        except queue.Full:
            cnt[CNT_DROPPED] += len(sel)


# ============================= BEAMFORMER =============================
class Beamformer:
    def __init__(self, positions):
        self.pos = np.asarray(positions, dtype=float)
        self.M = len(self.pos)
        self.win = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT))
        self.freqs = np.fft.rfftfreq(NFFT, 1.0 / SAMPLE_RATE)
        self.F = len(self.freqs)
        self.band = np.flatnonzero((self.freqs >= FREQ_MIN) & (self.freqs <= FREQ_MAX))
        self.hp = self.freqs < HIGH_PASS_HZ
        self.R = np.tile(np.eye(self.M) * 1e-6, (self.F, 1, 1)).astype(complex)
        self.ola = np.zeros(NFFT)
        self._az = None
        self._a = None
        self._w = None
        self._wkey = None
        self._hops = 0

    def steering(self, az_deg):
        """a[f, m]: phase of a plane wave from azimuth az at mic m (mics nearer the source hear it earlier)."""
        if az_deg != self._az:
            r = np.radians(az_deg)
            u = np.array([np.sin(r), np.cos(r)])
            tau = self.pos @ u / SPEED_OF_SOUND
            self._a = np.exp(1j * 2 * np.pi * self.freqs[:, None] * tau[None, :])
            self._az = az_deg
        return self._a

    def process(self, frame, az_deg, mode):
        """frame: (NFFT, M) float -> HOP output samples."""
        X = np.fft.rfft(frame * self.win[:, None], axis=0)                 # (F, M)
        self.R = ALPHA * self.R + (1 - ALPHA) * (X[:, :, None] * X[:, None, :].conj())

        a = self.steering(az_deg)
        if mode == 'mvdr' and self.M > 1 and len(self.band):
            key = (az_deg, mode)
            self._hops += 1
            if self._w is None or self._wkey != key or self._hops % WEIGHT_EVERY == 0:
                w = a / self.M
                idx = self.band
                Rb = self.R[idx]
                tr = np.trace(Rb, axis1=1, axis2=2).real / self.M
                Rb = Rb + (LOADING * tr + 1e-10)[:, None, None] * np.eye(self.M)
                ab = a[idx]
                Ria = np.linalg.solve(Rb, ab[..., None])[..., 0]            # R^-1 a
                den = np.einsum('fm,fm->f', ab.conj(), Ria).real
                w = w.copy()
                w[idx] = Ria / den[:, None]                                 # distortionless in look direction
                self._w = w
                self._wkey = key
            w = self._w
        else:
            w = a / self.M                                                  # delay-and-sum weights

        Y = np.einsum('fm,fm->f', w.conj(), X)
        Y[self.hp] = 0.0
        y = np.fft.irfft(Y, NFFT) * self.win
        self.ola += y
        out = self.ola[:HOP].copy()
        self.ola = np.concatenate([self.ola[HOP:], np.zeros(HOP)])
        return out


# ================================ THREADS =============================
def dsp_loop(q, player):
    global running
    bf = Beamformer(MIC_POSITIONS)
    in_buf = np.zeros((0, bf.M))
    while running:
        try:
            arr = q.get(timeout=0.2)
        except queue.Empty:
            continue
        in_buf = np.concatenate([in_buf, arr])
        outs = []
        while len(in_buf) >= NFFT:
            t0 = time.perf_counter()
            outs.append(bf.process(in_buf[:NFFT], AZIMUTH_SIGN * azimuth, mode))
            dt = time.perf_counter() - t0
            with lock:
                dsp_t['busy'] += dt
                dsp_t['calls'] += 1
                if dt > dsp_t['max']:
                    dsp_t['max'] = dt
            in_buf = in_buf[HOP:]
        if outs:
            raw = np.concatenate(outs)
            with lock:
                level['sum_sq'] += float(np.dot(raw, raw))
                level['n'] += len(raw)
                level['last_rms'] = math.sqrt(float(np.mean(raw * raw)))
            y = np.clip(raw * gain, -1.0, 1.0)
            try:
                player.stdin.write((y * 32767).astype('<i2').tobytes())
                player.stdin.flush()
            except (BrokenPipeError, ValueError):
                print("\naplay closed - stopping (check ALSA_DEVICE / `aplay -l`)")
                running = False


def aplay_stderr_loop(proc):
    """Count ALSA underruns. A non-zero count means the DSP is not keeping up with playback."""
    for line in proc.stderr:
        n = line.count(b'underrun')
        if n:
            with lock:
                stats['underrun'] += n


def stats_loop(cnt):
    last_t, last_n, last_busy = time.time(), 0, 0.0
    budget_ms = HOP / SAMPLE_RATE * 1000
    while running:
        time.sleep(5.0)
        now = time.time()
        n = cnt[CNT_FRAMES]
        with lock:
            busy = dsp_t['busy']
            mx = dsp_t['max']
            dsp_t['max'] = 0.0
            rms = level['last_rms']
            under = stats['underrun']
        wall = now - last_t
        load = (busy - last_busy) / wall * 100
        print(f"[stats] {(n - last_n) / wall:7.0f} frames/s | dsp {load:4.1f}% | "
              f"max call {mx * 1000:4.1f} ms (budget {budget_ms:.1f}) | "
              f"dropped {cnt[CNT_DROPPED]} | resync {cnt[CNT_RESYNC]} | underrun {under} | "
              f"rms {rms:.4f} | az {azimuth:+.0f} | {mode} | gain {gain:g}")
        last_t, last_n, last_busy = now, n, busy


def sweep(step=15, hold=2.0):
    """Sweep the beam across -90..+90 and report output level per angle.
    Leaves the beam pointed at the loudest angle."""
    global azimuth
    print(f"sweeping -90..+90 in {step} deg steps, {hold:g} s each. Keep the source still.")
    results = []
    for ang in range(-90, 91, step):
        azimuth = float(ang)
        time.sleep(0.5)                        # let the covariance settle
        with lock:
            level['sum_sq'] = 0.0
            level['n'] = 0
        time.sleep(hold)
        with lock:
            s, n = level['sum_sq'], level['n']
        rms = math.sqrt(s / n) if n else 0.0
        db = 20 * math.log10(rms) if rms > 0 else -120.0
        results.append((ang, rms))
        print(f"  {ang:+4d} deg   rms {rms:.5f}   {db:6.1f} dBFS")
    peak_ang, _ = max(results, key=lambda r: r[1])
    azimuth = float(peak_ang)
    print(f"peak at {peak_ang:+d} deg, beam set there.")
    print("If that is not where the source is, flip AZIMUTH_SIGN.")


# ================================ MAIN ================================
def main():
    global running, azimuth, mode, gain
    port = sys.argv[1] if len(sys.argv) > 1 else SERIAL_PORT

    d = [np.linalg.norm(MIC_POSITIONS[i] - MIC_POSITIONS[j])
         for i in range(len(MIC_POSITIONS)) for j in range(i + 1, len(MIC_POSITIONS))]
    print("=" * 66)
    print("WYSH beamformer")
    print(f"  serial   : {port} @ {SERIAL_BAUD}")
    print(f"  channels : {MIC_CHANNELS}  ({len(MIC_CHANNELS)} mics)")
    if d:
        print(f"  spacing  : {min(d) * 1000:.1f} mm min  -> spatial aliasing above ~{SPEED_OF_SOUND / (2 * min(d)):.0f} Hz")
    print(f"  audio out: {ALSA_DEVICE}  (buffer {ALSA_BUFFER_US / 1000:.0f} ms)")
    print("=" * 66)

    q = mp.Queue(maxsize=QUEUE_CHUNKS)
    cnt = mp.Array('q', 3)
    reader = mp.Process(target=reader_proc, args=(port, SERIAL_BAUD, q, cnt), daemon=True)
    reader.start()
    time.sleep(0.5)
    if not reader.is_alive():
        print("Reader could not open the serial port. Check the message above.")
        print("Check: ls -l /dev/serial0  |  python3 -m serial.tools.list_ports -v")
        sys.exit(1)

    player = subprocess.Popen(
        ['aplay', '-D', ALSA_DEVICE, '-t', 'raw', '-f', 'S16_LE',
         '-r', str(int(round(SAMPLE_RATE))), '-c', '1', '-B', str(ALSA_BUFFER_US)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    threads = [
        (dsp_loop, (q, player)),
        (stats_loop, (cnt,)),
        (aplay_stderr_loop, (player,)),
    ]
    for fn, args in threads:
        threading.Thread(target=fn, args=args, daemon=True).start()

    print(__doc__.split("Controls while running:")[1])
    while running:
        try:
            cmd = input("beam> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break
        if not cmd:
            continue
        if cmd == 'q':
            break
        if cmd in ('s', 'sweep'):
            sweep()
        elif cmd == 'm':
            mode = 'das' if mode == 'mvdr' else 'mvdr'
            print(f"mode: {mode}")
        elif cmd.startswith('g'):
            try:
                gain = float(cmd[1:])
                print(f"gain: {gain:g}")
            except ValueError:
                print("usage: g 5")
        else:
            try:
                a = float(cmd)
                if -180 <= a <= 180:
                    azimuth = a
                    print(f"beam -> {a:+.0f} deg")
                else:
                    print("angle must be -180..180")
            except ValueError:
                print("unknown command")

    running = False
    time.sleep(0.3)
    try:
        player.stdin.close()
    except Exception:
        pass
    player.terminate()
    reader.terminate()
    print("done")


if __name__ == '__main__':
    main()
