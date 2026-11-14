"""Clean a raw ECoG recording: channel selection, filtering, re-referencing."""
import numpy as np

LINE_FREQ = 60.0             # US power line
L_FREQ, H_FREQ = 0.5, 200.0  # band-pass range (Hz)


def select_channels(raw, channels, auto_reject=True):
    """Keep only channels listed as ECOG and 'good' in channels.tsv.

    Drops EKG, DC trigger lines, duplicate $G11/$G12, N/A, and clinically
    marked bad contacts. Optionally also drops channels that are flat or
    far noisier than the rest (std < median/10 or > median*10).
    Modifies raw in place and returns it.
    """
    keep = channels.query("type == 'ECOG' and status == 'good'")["name"].tolist()
    dropped = sorted(set(raw.ch_names) - set(keep))
    raw.pick(keep)
    raw.set_channel_types({ch: "ecog" for ch in raw.ch_names})
    print(f"Kept {len(keep)} ECoG channels; dropped {len(dropped)}: {dropped}")

    if auto_reject:
        std = raw.get_data().std(axis=1)
        med = np.median(std)
        auto_bad = [ch for ch, s in zip(raw.ch_names, std) if s < med / 10 or s > med * 10]
        if auto_bad:
            print("Auto-flagged noisy/flat channels (dropping):", auto_bad)
            raw.drop_channels(auto_bad)
    return raw


def filter_signal(raw, line_freq=LINE_FREQ, l_freq=L_FREQ, h_freq=H_FREQ):
    """Notch out line noise + harmonics, then band-pass. In place; returns raw."""
    raw.notch_filter(np.arange(line_freq, h_freq + 1, line_freq), verbose="error")
    raw.filter(l_freq, h_freq, verbose="error")
    return raw


def rereference(raw):
    """Common average reference (subtract mean of all channels). In place.

    Bipolar referencing between neighbouring contacts is a common alternative.
    """
    raw.set_eeg_reference("average", ch_type="ecog", verbose="error")
    return raw


def clean(raw, channels):
    """Full cleaning pipeline: select channels -> filter -> re-reference."""
    select_channels(raw, channels)
    filter_signal(raw)
    rereference(raw)
    return raw
