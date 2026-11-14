"""Cut a cleaned recording into labelled windows and compute features.

Run the whole pipeline (load -> clean -> window -> features -> save):
    py src/segmentation.py
Output: data/processed/<base>_clean.npz
"""
from pathlib import Path

import mne
import numpy as np

WIN_SEC = 2.0     # window length
STEP_SEC = 1.0    # hop between windows (1 s overlap)
MARGIN_SEC = 5.0  # drop windows this close to onset/offset (ambiguous labels)

BANDS = {         # frequency bands for power features (Hz)
    "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
    "beta": (13, 30), "gamma": (30, 80), "high_gamma": (80, 200),
}

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def make_windows(raw, onset, offset, win_sec=WIN_SEC, step_sec=STEP_SEC, margin_sec=MARGIN_SEC):
    """Return X (windows, channels, samples), y (1 = seizure), window start times.

    Windows fully inside [onset+margin, offset-margin] are seizure (1);
    windows fully outside [onset-margin, offset+margin] are non-seizure (0);
    windows near a boundary are skipped.
    """
    sfreq = raw.info["sfreq"]
    data = raw.get_data()
    win, step = int(win_sec * sfreq), int(step_sec * sfreq)
    starts = np.arange(0, data.shape[1] - win + 1, step)

    X, y, t0 = [], [], []
    for s in starts:
        a, b = s / sfreq, (s + win) / sfreq
        if onset + margin_sec <= a and b <= offset - margin_sec:
            label = 1
        elif b <= onset - margin_sec or a >= offset + margin_sec:
            label = 0
        else:
            continue
        X.append(data[:, s:s + win])
        y.append(label)
        t0.append(a)

    X, y, t0 = np.stack(X).astype(np.float32), np.array(y), np.array(t0)
    print(f"{len(y)} windows: {int(y.sum())} seizure, {int((y == 0).sum())} non-seizure "
          f"(skipped {len(starts) - len(y)} near boundaries)")
    return X, y, t0


def compute_features(X, sfreq, bands=BANDS):
    """Log band power per channel per band + log line length per channel.

    Returns (windows, channels * len(bands) + channels).
    """
    fmax = max(hi for _, hi in bands.values())
    psd, freqs = mne.time_frequency.psd_array_welch(
        X, sfreq, fmin=1, fmax=fmax, n_fft=int(sfreq), verbose="error")
    band_power = np.stack(
        [psd[:, :, (freqs >= lo) & (freqs < hi)].mean(axis=-1) for lo, hi in bands.values()],
        axis=-1)
    log_bp = np.log10(band_power + 1e-30)
    line_length = np.abs(np.diff(X, axis=-1)).sum(axis=-1)
    return np.concatenate([log_bp.reshape(len(X), -1), np.log10(line_length + 1e-30)], axis=1)


if __name__ == "__main__":
    from data_loader import DEFAULT_BASE, load_recording, seizure_times
    from pre_process import clean

    raw, channels, events = load_recording(DEFAULT_BASE)
    onset, offset = seizure_times(events)
    print(f"Seizure: {onset:.2f} s -> {offset:.2f} s ({offset - onset:.1f} s)")

    clean(raw, channels)
    X, y, t0 = make_windows(raw, onset, offset)
    features = compute_features(X, raw.info["sfreq"])
    print("Feature matrix:", features.shape)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{DEFAULT_BASE}_clean.npz"
    np.savez_compressed(
        out, X=X, y=y, window_start=t0, features=features,
        ch_names=np.array(raw.ch_names), band_names=np.array(list(BANDS)),
        sfreq=raw.info["sfreq"])
    print("Saved", out.relative_to(PROCESSED_DIR.parent.parent))
