# side-splitter

Finds, labels and trims the sides of a single-take vinyl recording in Audacity (3.7.x, Windows 11),
driven through Audacity's `mod-script-pipe`. Handles any number of sides (needle drops and lifts
mid-recording).

## How it works

The recording is modelled as three levels: phono-stage hiss (arm up) < groove noise < music.

1. Band-pass (100 Hz–8 kHz) short-time RMS envelope, median-smoothed to remove pops, needle
   thumps and lead-out clicks.
2. Coarse music regions above an automatic threshold, grouped into sides; a new side starts after a
   gap > 12 s or a gap containing ≥ 2 s of arm-up hiss.
3. Each side's start/end is refined against its own lead-in/lead-out groove floor (with hysteresis
   to follow fade-ins and decays), then padded.

## Setup

```
pip install -r requirements.txt
```

In Audacity: Edit → Preferences → Modules → **mod-script-pipe** = Enabled, then restart Audacity.

## Usage (recording open in Audacity)

```
python vinyl_trim.py labels --plot   # non-destructive: adds "Side A/B/..." labels, saves a diagnostic plot
python vinyl_trim.py trim            # fades, removes lead-ins/outs and flip gaps, relabels (asks first)
python vinyl_trim.py analyze rec.flac --plot   # offline analysis, no Audacity needed
```

Then File → Export Audio → *Multiple files*, split by labels, for one file per side.

Tuning options: `--music-db`, `--track-gap`, `--min-side`, `--pad-start`, `--pad-end`,
`--fade-in`, `--fade-out`, `--keep-gap`.

## Notes

- If Audacity shows a "mixing down" warning during the internal export, disable it under
  Preferences → Warnings.
- Save the project before `trim`; each edit is a separate undo step.
