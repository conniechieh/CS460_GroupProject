%pip install PyQt6


%matplotlib qt
import mne
#
# Point to the .vhdr file (keep .eeg and .vmrk in the same folder) (make sure to change name to match where your file is downloaded)
raw = mne.io.read_raw_brainvision("C:\Users\jjarb\Downloads\sub-pt01_ses-presurgery_task-ictal_acq-ecog_run-01_ieeg (2).eeg", preload=True)


# Print basic metadata (channels, sampling rate, duration)
print(raw.info)


# Open an interactive plot to scroll through channels and time
raw.plot(duration=10, n_channels=20, scalings="auto", block=True)