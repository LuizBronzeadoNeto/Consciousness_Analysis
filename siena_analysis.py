import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from datetime import datetime
def load_data():
    raw = mne.io.read_raw_edf("physionet_siena_eeg/PhysioNet_Siena/PN00/PN00-1.edf", preload=True)
    eeg_channels = [ch for ch in raw.ch_names if ch.startswith("EEG")]
    raw.pick(eeg_channels)
    raw.filter(l_freq = 0.5, h_freq = 45.0)
    raw.notch_filter(freqs = 50.0)
    time_format = "%H.%M.%S"
    reg_start = datetime.strptime("19.39.33", time_format)
    sz_start = datetime.strptime("19.58.36", time_format)
    sz_end = datetime.strptime("19.59.46", time_format)
    onset_sec = (sz_start - reg_start).total_seconds()
    duration_sec = (sz_end - sz_start).total_seconds()

    seizure_annotation = mne.Annotations(onset=[onset_sec], duration=[duration_sec], description=['Seizure'])
    raw.set_annotations(seizure_annotation)
    return raw


def main():
    raw = load_data()
    fig = raw.plot(duration=15.0, 
               n_channels=20, 
               scalings='auto', 
               title="Filtered EEG - Scroll to see the Seizure",
               block=True)
    epochs = mne.make_fixed_length_epochs(raw, duration = 2.0, preload = True)
    psd = epochs.compute_psd(method = 'multiaper', fmin = 0.5, fmax = 14.0)
    psds, freqs = psd.get_data(return_freqs=True)

    bands = {
        "Delta": {0.5, 4.0},
        "Alpha": {8.0, 13.0}
    }
    epoch_data = []
    epoch_data = epochs.events
    event_id_reverse = {v: k for k, v in epochs.events_id.items()}
    sampling_rate = raw.info('sfreq')


if __name__ == "__main__":
    main()
    

