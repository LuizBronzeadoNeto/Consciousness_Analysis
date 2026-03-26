import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from datetime import datetime
import os
import shutil
def load_data(path):
    raw = mne.io.read_raw_edf(path, preload=True)
    eeg_channels = [ch for ch in raw.ch_names if ch.startswith("EEG")]
    raw.pick(eeg_channels)
    raw.filter(l_freq = 0.5, h_freq = 45.0) #no idea if there are more filters could be applied without messing with seizure detection
    raw.notch_filter(freqs = 50.0)
    return raw

def discretize_raw_data(raw, patient_num, collect_num): #I don't know what I'm doing
    epochs = mne.make_fixed_length_epochs(raw, duration = 2.0, preload = True)
    psd = epochs.compute_psd(method = 'multitaper', fmin = 0.5, fmax = 14.0)
    psds, freqs = psd.get_data(return_freqs=True)

    bands = {
        "Delta": {0.5, 4.0},
        "Alpha": {8.0, 13.0}
    }
    epoch_data = []
    events = epochs.events
    event_id_reverse = {v: k for k, v in epochs.event_id.items()}
    sampling_rate = raw.info['sfreq']
    for i in range(len(epochs)):
        label = event_id_reverse.get(events[i, 2], "Normal")
        t_sec = events[i, 0] / sampling_rate
        row = {'t': t_sec, 'l': label}
        for ch_index, ch_name in enumerate(epochs.ch_names):
            for band_name, (fmin, fmax) in bands.items():
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                raw_power = np.mean(psds[i, ch_index, freq_mask])
                power_db = 10 * np.log10(raw_power + 1e-10)

                row[f"{ch_name}_{band_name}.dB"] = power_db
        epoch_data.append(row)
    
    df = pd.DataFrame(epoch_data)
    df.to_csv(f"PN0{patient_num}-{collect_num}_delta_alpha.csv", index = False)
    shutil.move(f"PN0{patient_num}-{collect_num}_delta_alpha.csv", f"siena_discretized/PN0{patient_num}")

def main():
    path = "physionet_siena_eeg/PhysioNet_Siena/PN00/PN00-1.edf"
    raw = load_data(path)
    time_format = "%H.%M.%S"
    reg_start = datetime.strptime("19.39.33", time_format)
    sz_start = datetime.strptime("19.58.36", time_format)
    sz_end = datetime.strptime("19.59.46", time_format)
    onset_sec = (sz_start - reg_start).total_seconds()
    duration_sec = (sz_end - sz_start).total_seconds()

    seizure_annotation = mne.Annotations(onset=[onset_sec], duration=[duration_sec], description=['Seizure'])
    raw.set_annotations(seizure_annotation)
    fig = raw.plot(duration=15.0, 
               n_channels=20, 
               scalings='auto', 
               title="Filtered EEG - Scroll to see the Seizure",
               block=True)
    discretize_raw_data(raw, 0, 1)


if __name__ == "__main__":
    main()
    

