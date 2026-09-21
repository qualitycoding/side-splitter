#!/usr/bin/env python3
"""
vinyl_trim.py - find, label and trim the sides of a single-take vinyl recording
in Audacity (tested design target: Audacity 3.7.x on Windows 11).

Signal model (per side):
    arm up (phono hiss) -> needle drop -> lead-in groove -> music
    -> lead-out groove (clicks every ~1.8 s) -> needle lift -> arm up ...
Three levels are used: hiss < groove noise < music.

Commands (Audacity open with the recording, mod-script-pipe enabled):
    python vinyl_trim.py labels            add a "Side A/B/..." label over each side (non-destructive)
    python vinyl_trim.py trim              fade, remove lead-ins/outs and flip gaps, then label
    python vinyl_trim.py analyze rec.flac  offline analysis of a file; Audacity not needed

Add --plot to save a diagnostic plot (vinyl_trim_plot.png) for tuning.
Requires: pip install numpy scipy soundfile   (matplotlib only for --plot)
"""

import argparse
import json
import os
import string
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfilt


# --------------------------------------------------------------------------- parameters

@dataclass
class Params:
    hop: float = 0.05                 # envelope frame length (s)
    band_lo: float = 100.0            # band-pass (Hz): removes rumble and arm thumps
    band_hi: float = 8000.0
    median_s: float = 0.5             # median smoothing: kills pops, drop/lift and lead-out clicks
    music_db: Optional[float] = None  # absolute music threshold (dBFS); None = automatic
    music_below_ref_db: float = 20.0  # automatic threshold = typical music level minus this
    min_music_s: float = 1.5          # music must persist at least this long
    track_gap_s: float = 12.0         # a gap between music longer than this = side change
    lift_s: float = 2.0               # this much arm-up hiss inside a gap = side change
    hiss_margin_db: float = 4.0       # "arm up" = within this of the hiss floor
    start_margin_db: float = 6.0      # onset = first point this far above lead-in groove floor...
    start_extend_db: float = 2.0      # ...then back further while still this far above (fade-ins)
    end_margin_db: float = 2.0        # end = where decay reaches this far above lead-out floor
    floor_window_s: float = 5.0       # groove-floor estimation window
    max_extend_s: float = 15.0        # max onset/decay refinement
    min_side_s: float = 60.0          # discard shorter "sides" (cueing, handling noise)
    pad_start_s: float = 0.3
    pad_end_s: float = 0.5
    fade_in_s: float = 0.25
    fade_out_s: float = 1.5
    keep_gap_s: float = 2.0           # silence left between sides in trim mode


# --------------------------------------------------------------------------- analysis

def band_envelope_db(path, p):
    """Band-limited short-time RMS in dBFS, median-smoothed. Streams the file in blocks."""
    info = sf.info(path)
    sr = info.samplerate
    hop = int(round(p.hop * sr))
    sos = butter(4, [p.band_lo, min(p.band_hi, 0.45 * sr)], btype="bandpass", fs=sr, output="sos")
    zi = np.zeros((sos.shape[0], 2))
    carry = np.zeros(0)
    chunks = []
    for block in sf.blocks(path, blocksize=hop * 400, dtype="float32", always_2d=True):
        y, zi = sosfilt(sos, block.mean(axis=1, dtype=np.float64), zi=zi)
        y = np.concatenate([carry, y])
        n = len(y) // hop
        if n:
            f = y[: n * hop].reshape(n, hop)
            chunks.append(np.sqrt(np.mean(f * f, axis=1)))
        carry = y[n * hop:]
    rms = np.concatenate(chunks) if chunks else np.zeros(1)
    db = 20 * np.log10(np.maximum(rms, 1e-7))
    k = max(3, int(round(p.median_s / p.hop)) | 1)
    return median_filter(db, size=k, mode="nearest"), info.frames / sr


def runs(mask):
    """(start, end_exclusive) index pairs of True runs."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def groove_floor(seg, hiss_limit):
    seg = seg[seg > hiss_limit]  # needle in the groove, not arm up
    return float(np.median(seg)) if seg.size >= 10 else None


def detect_sides(db, duration, p):
    fr = lambda s: int(round(s / p.hop))
    n = len(db)
    hiss_db = float(np.percentile(db, 1))
    hiss_limit = hiss_db + p.hiss_margin_db
    loud = db[db > hiss_limit + 10]
    ref = float(np.median(loud)) if loud.size else float(np.median(db))
    t_music = p.music_db if p.music_db is not None else max(ref - p.music_below_ref_db, hiss_limit + 6)
    diag = dict(hiss_db=hiss_db, hiss_limit=hiss_limit, ref_db=ref, music_db=t_music)

    # 1. coarse music regions
    regions = [(a, b) for a, b in runs(db > t_music) if b - a >= fr(p.min_music_s)]
    if not regions:
        return [], diag

    # 2. group into sides: split on long gaps or on arm-up hiss inside a gap
    hiss = db < hiss_limit
    groups = [list(regions[0])]
    for a, b in regions[1:]:
        gs = groups[-1][1]
        lifted = any(y - x >= fr(p.lift_s) for x, y in runs(hiss[gs:a]))
        if a - gs > fr(p.track_gap_s) or lifted:
            groups.append([a, b])
        else:
            groups[-1][1] = b
    groups = [g for g in groups if g[1] - g[0] >= fr(p.min_side_s)]

    # 3. refine each side against its local lead-in / lead-out groove floor
    one, win, ext = fr(1.0), fr(p.floor_window_s), fr(p.max_extend_s)
    sides = []
    for a, b in groups:
        f0 = groove_floor(db[max(0, a - one - win): max(0, a - one)], hiss_limit)
        s = a
        if f0 is not None:
            lim = max(0, a - ext)
            while s > lim and db[s - 1] > f0 + p.start_margin_db:
                s -= 1
            while s > lim and db[s - 1] > f0 + p.start_extend_db and db[s - 1] < db[s] + 0.5:
                s -= 1  # hysteresis: follow a fade-in down towards the groove floor
        f1 = groove_floor(db[min(n, b + one): min(n, b + one + win)], hiss_limit)
        e = b
        if f1 is not None:
            lim = min(n, b + ext)
            while e < lim and db[e] > f1 + p.end_margin_db:
                e += 1
        st = max(0.0, s * p.hop - p.pad_start_s)
        en = min(duration, e * p.hop + p.pad_end_s)
        if sides:
            st = max(st, sides[-1]["end"])
        sides.append(dict(start=st, end=en, floor_in=f0, floor_out=f1))
    return sides, diag


def side_names(k):
    return [f"Side {string.ascii_uppercase[i]}" if i < 26 else f"Side {i + 1}" for i in range(k)]


def fmt(t):
    m, s = divmod(t, 60)
    return f"{int(m):3d}:{s:05.2f}"


def report(sides, diag, duration):
    print(f"\nRecording {fmt(duration)}   hiss {diag['hiss_db']:.1f} dBFS   "
          f"music level {diag['ref_db']:.1f}   music threshold {diag['music_db']:.1f}")
    if not sides:
        print("No sides found. Try --plot and --music-db.")
        return
    print(f"{'':8} {'start':>9} {'end':>9} {'length':>9}   lead-in / lead-out floor (dBFS)")
    for name, s in zip(side_names(len(sides)), sides):
        fi = "  n/a" if s["floor_in"] is None else f"{s['floor_in']:6.1f}"
        fo = "  n/a" if s["floor_out"] is None else f"{s['floor_out']:6.1f}"
        print(f"{name:8} {fmt(s['start'])} {fmt(s['end'])} {fmt(s['end'] - s['start'])}   {fi} / {fo}")


def save_plot(db, sides, diag, p, out="vinyl_trim_plot.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(len(db)) * p.hop / 60
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(t, db, lw=0.4, color="k")
    ax.axhline(diag["hiss_limit"], ls=":", color="tab:blue", label="arm-up limit")
    ax.axhline(diag["music_db"], ls="--", color="tab:red", label="music threshold")
    for name, s in zip(side_names(len(sides)), sides):
        ax.axvspan(s["start"] / 60, s["end"] / 60, color="tab:green", alpha=0.15)
        ax.text(s["start"] / 60, diag["ref_db"] + 8, name, fontsize=9)
    ax.set_xlabel("minutes")
    ax.set_ylabel("band RMS (dBFS)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"Plot saved to {os.path.abspath(out)}")


# --------------------------------------------------------------------------- Audacity pipe

class Audacity:
    def __init__(self):
        if sys.platform == "win32":
            to_name, from_name, self.eol = r"\\.\pipe\ToSrvPipe", r"\\.\pipe\FromSrvPipe", "\r\n\0"
        else:
            uid = os.getuid()
            to_name = f"/tmp/audacity_script_pipe.to.{uid}"
            from_name = f"/tmp/audacity_script_pipe.from.{uid}"
            self.eol = "\n"
        if not os.path.exists(to_name):
            sys.exit("Audacity pipe not found. Start Audacity and enable mod-script-pipe "
                     "(Edit > Preferences > Modules), then restart Audacity.")
        self.to = open(to_name, "w")
        self.frm = open(from_name, "r")

    def cmd(self, command):
        self.to.write(command + self.eol)
        self.to.flush()
        lines = []
        while True:
            line = self.frm.readline()
            if line == "":
                raise RuntimeError("Audacity closed the pipe")
            if line == "\n" and lines:
                break
            lines.append(line)
        resp = "".join(lines)
        if "BatchCommand finished: OK" not in resp:
            raise RuntimeError(f"Audacity command failed: {command}\n{resp}")
        return resp

    def info(self, kind):
        resp = self.cmd(f"GetInfo: Type={kind} Format=JSON")
        body = resp[: resp.rfind("BatchCommand finished")].strip()
        return json.loads(body) if body else []

    def project_end(self):
        return max((t.get("end", 0.0) for t in self.info("Tracks")), default=0.0)

    def labels(self):
        out = []
        for _track, items in self.info("Labels"):
            out.extend(items)
        return out

    def select_time(self, a, b):
        self.cmd(f"SelectTime: Start={a:.6f} End={b:.6f} RelativeTo=ProjectStart")

    def export_analysis_copy(self, path):
        end = self.project_end()
        self.cmd("SelectAll:")
        self.select_time(0.0, end)
        self.cmd(f'Export2: Filename="{path.replace(os.sep, "/")}" NumChannels=2')
        return end

    def add_labels(self, spans, names):
        for (a, b), name in zip(spans, names):
            self.select_time(a, b)
            self.cmd("AddLabel:")
            for i, (s, _e, text) in enumerate(self.labels()):
                if abs(s - a) < 1e-3 and text == "":
                    self.cmd(f'SetLabel: Label={i} Text="{name}"')
                    break


def trim_project(aud, sides, total, p):
    """Destructive: edits right-to-left so earlier positions stay valid."""
    spans = [(s["start"], s["end"]) for s in sides]
    aud.cmd("SelectAll:")
    if total - spans[-1][1] > 1e-3:
        aud.select_time(spans[-1][1], total)
        aud.cmd("Delete:")
    for i in range(len(spans) - 1, -1, -1):
        st, en = spans[i]
        length = en - st
        aud.select_time(en - min(p.fade_out_s, length / 4), en)
        aud.cmd("FadeOut:")
        aud.select_time(st, st + min(p.fade_in_s, length / 4))
        aud.cmd("FadeIn:")
        prev_end = spans[i - 1][1] if i else 0.0
        gap = st - prev_end
        keep = min(p.keep_gap_s, gap) if i else 0.0
        if keep > 1e-3:
            aud.select_time(prev_end, prev_end + keep)
            aud.cmd("Silence:")
        if gap - keep > 1e-3:
            aud.select_time(prev_end + keep, st)
            aud.cmd("Delete:")
    new, t = [], 0.0
    for i, (st, en) in enumerate(spans):
        if i:
            t += min(p.keep_gap_s, st - spans[i - 1][1])
        new.append((t, t + en - st))
        t += en - st
    return new


# --------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="Label or trim vinyl sides in Audacity.")
    ap.add_argument("mode", choices=["labels", "trim", "analyze"])
    ap.add_argument("file", nargs="?", help="audio file (analyze mode only)")
    ap.add_argument("--plot", action="store_true", help="save vinyl_trim_plot.png")
    ap.add_argument("--yes", action="store_true", help="trim without asking")
    ap.add_argument("--music-db", type=float, help="absolute music threshold, dBFS")
    ap.add_argument("--track-gap", type=float, help="max gap between tracks within a side (s)")
    ap.add_argument("--min-side", type=float, help="shortest side kept (s)")
    ap.add_argument("--pad-start", type=float)
    ap.add_argument("--pad-end", type=float)
    ap.add_argument("--fade-in", type=float)
    ap.add_argument("--fade-out", type=float)
    ap.add_argument("--keep-gap", type=float, help="silence left between sides when trimming (s)")
    a = ap.parse_args()

    p = Params()
    for arg, field in [("music_db", "music_db"), ("track_gap", "track_gap_s"),
                       ("min_side", "min_side_s"), ("pad_start", "pad_start_s"),
                       ("pad_end", "pad_end_s"), ("fade_in", "fade_in_s"),
                       ("fade_out", "fade_out_s"), ("keep_gap", "keep_gap_s")]:
        v = getattr(a, arg)
        if v is not None:
            setattr(p, field, v)

    if a.mode == "analyze":
        if not a.file:
            ap.error("analyze needs a file")
        db, duration = band_envelope_db(a.file, p)
        sides, diag = detect_sides(db, duration, p)
        report(sides, diag, duration)
        if a.plot:
            save_plot(db, sides, diag, p)
        return

    aud = Audacity()
    tmp = os.path.join(tempfile.gettempdir(), "vinyl_trim_analysis.flac")
    print("Exporting analysis copy from Audacity...")
    total = aud.export_analysis_copy(tmp)
    try:
        db, duration = band_envelope_db(tmp, p)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    sides, diag = detect_sides(db, duration, p)
    report(sides, diag, duration)
    if a.plot:
        save_plot(db, sides, diag, p)
    if not sides:
        return
    names = side_names(len(sides))

    if a.mode == "labels":
        aud.add_labels([(s["start"], s["end"]) for s in sides], names)
        print("Labels added. Check them, then run 'trim' or use Export Audio > Multiple files.")
        return

    if not a.yes and input("\nTrim the project as above? Save first. [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return
    new_spans = trim_project(aud, sides, total, p)
    aud.add_labels(new_spans, names)
    print("Done. Sides are labelled; Export Audio > Multiple files will split them.")


if __name__ == "__main__":
    main()
