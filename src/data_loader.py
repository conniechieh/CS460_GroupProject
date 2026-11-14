"""Load a BrainVision iEEG recording plus its BIDS sidecar files.

Expected layout (data/ is git-ignored, download from OpenNeuro ds003029):
    data/raw/<base>_ieeg.vhdr   header  (references the .eeg and .vmrk by name,
    data/raw/<base>_ieeg.eeg    data     so keep the original file names --
    data/raw/<base>_ieeg.vmrk   markers  no " (1)" download suffixes)
    data/raw/<base>_channels.tsv  channel types + good/bad status
    data/raw/<base>_events.tsv    seizure onset/offset annotations

Run directly to open an interactive plot:  py src/data_loader.py
"""
from pathlib import Path

import mne
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DEFAULT_BASE = "sub-pt01_ses-presurgery_task-ictal_acq-ecog_run-01"


def load_recording(base=DEFAULT_BASE, raw_dir=RAW_DIR):
    """Return (raw, channels, events) for one recording.

    raw      : mne.io.Raw with data loaded
    channels : DataFrame from <base>_channels.tsv
    events   : DataFrame from <base>_events.tsv
    """
    raw_dir = Path(raw_dir)
    raw = mne.io.read_raw_brainvision(raw_dir / f"{base}_ieeg.vhdr", preload=True, verbose="error")
    channels = pd.read_csv(raw_dir / f"{base}_channels.tsv", sep="\t")
    events = pd.read_csv(raw_dir / f"{base}_events.tsv", sep="\t")
    return raw, channels, events


def seizure_times(events):
    """Return (onset, offset) in seconds from an events table."""
    onset = events.loc[events.trial_type == "onset", "onset"].iloc[0]
    offset = events.loc[events.trial_type == "offset", "onset"].iloc[0]
    return float(onset), float(offset)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("QtAgg")  # interactive window (needs PyQt6)

    raw, channels, events = load_recording()
    onset, offset = seizure_times(events)
    print(raw.info)
    print(f"Duration: {raw.times[-1]:.1f} s, {len(raw.ch_names)} channels")
    print(f"Seizure: {onset:.2f} s -> {offset:.2f} s")

    # Arrow keys scroll time/channels, +/- change amplitude.
    raw.plot(duration=10, n_channels=20, scalings="auto", block=True)
