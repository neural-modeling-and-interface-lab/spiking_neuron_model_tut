#!/usr/bin/env python3
"""
Author: Chen Sun (Feb 19, 2026)
Picked from identify_stdp/prescreening/preprpoecessing_pipeline_siso_function.py
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import find_peaks
from diptest import diptest
import numpy as np



def plot_rate_heatmap(pkl_path: str, save_path: str, n_bins: int = 1000, event_list: list = None, neuron_list: list = None):
    """Plot heatmap: y=neurons (names on yticks), x=time bins, color=bin-wise firing rate (Hz).
    Below neurons, add one row per event type as a raster. Save to save_path.
    event_list: if provided, only show events whose name is in this list.
    neuron_list: if provided, only show neurons whose name is in this list."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    neurons = data.get("neurons", [])
    if neuron_list is not None:
        neurons = [n for n in neurons if n.get("name", "") in neuron_list]
    if not neurons:
        return
    # Get the first and last timestamp of the 'TRIAL' event
    trial_events = [e for e in data.get("events", []) if e.get("name") == "TRIAL" and e.get("timestamps")]
    if not trial_events or not trial_events[0]["timestamps"]:
        return
    tbeg = trial_events[0]["timestamps"][0]
    tend = trial_events[0]["timestamps"][-1]
    # tbeg = data['tbeg']
    # tend = data['tend']
    duration_sec = tend - tbeg if tend and tbeg is not None else 0
    if duration_sec <= 0:
        return

    events = data.get("events", [])
    events = [e for e in events if e.get("timestamps")]  # only events that have timestamps
    if event_list is not None:
        events = [e for e in events if e.get("name", "") in event_list]
    n_neurons = len(neurons)
    n_events = len(events)
    # Layout: top = neuron heatmap, bottom = event rasters (shared x)
    n_rows = 1 + (1 if n_events else 0)
    height_ratios = [max(6, n_neurons * 0.15), max(1.5, n_events * 0.25)] if n_events else [1]
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, sum(height_ratios)), height_ratios=height_ratios, sharex=True)

    bin_edges = np.linspace(tbeg, tend, n_bins + 1)
    bin_duration_sec = (tend - tbeg) / n_bins
    rate_matrix = np.zeros((n_neurons, n_bins))
    names = []
    for i, n in enumerate(neurons):
        names.append(n.get("name", f"neuron_{i}"))
        ts = np.array(n["timestamps"])
        counts, _ = np.histogram(ts, bins=bin_edges)
        rate_matrix[i, :] = counts / bin_duration_sec  # Hz

    ax_heat = axes[0] if n_events else axes
    im = ax_heat.imshow(rate_matrix, aspect="auto", interpolation="nearest",
                        extent=[tbeg, tend, n_neurons - 0.5, -0.5], origin="upper")
    ax_heat.set_yticks(np.arange(n_neurons))
    ax_heat.set_yticklabels(names, fontsize=6)
    ax_heat.set_ylabel("Neuron")
    ax_heat.set_title("Firing rate (Hz)")
    plt.colorbar(im, ax=ax_heat, label="Firing rate (Hz)")

    if n_events:
        ax_ev = axes[1]
        for j, ev in enumerate(events):
            ts = np.array(ev.get("timestamps", []))
            name = ev.get("name", f"event_{j}")
            ts = ts[(ts >= tbeg) & (ts <= tend)]
            ax_ev.scatter(ts, np.ones_like(ts) * j, c="k", s=1, marker="|")
        ax_ev.set_yticks(np.arange(n_events))
        ax_ev.set_yticklabels([e.get("name", f"ev_{i}") for i, e in enumerate(events)], fontsize=6)
        ax_ev.set_ylim(-0.5, n_events - 0.5)
        ax_ev.set_ylabel("Event")
        ax_ev.set_title("Event rasters")

    ax_heat.set_xlabel("Time (s)")
    plt.tight_layout()
    # Align y-axes: same left edge and width for both panels
    if n_events:
        pos_heat = ax_heat.get_position()
        pos_ev = ax_ev.get_position()
        ax_ev.set_position([pos_heat.x0, pos_ev.y0, pos_heat.width, pos_ev.height])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")

    plt.show()
    plt.close()
    return neurons


def plot_all_neurons_silent_periods(neurons, save_dir, sample_rate):
    neuron_spike_trains = {}
    global_max_time = 0
    for neuron_id, spike_times in neurons.items():
        spike_times = np.array(spike_times) / sample_rate * 1000
        spike_indices = np.round(spike_times).astype(int)
        global_max_time = max(global_max_time, int(np.max(spike_indices)) if len(spike_indices) > 0 else 0)
        spike_train = np.zeros(global_max_time + 1)
        spike_train[spike_indices] = 1
        neuron_spike_trains[neuron_id] = spike_train

    silent_periods = {}
    for neuron_id, spike_train in neuron_spike_trains.items():
        spike_indices = np.where(spike_train == 1)[0]
        if len(spike_indices) == 0:
            silent_periods[neuron_id] = [(0, global_max_time)]
            continue
        isis = np.diff(np.concatenate([spike_indices, [global_max_time]]))
        top_3_indices = np.argsort(isis)[-3:][::-1]
        silent_periods[neuron_id] = [(spike_indices[idx] if idx != len(spike_indices)-1 else spike_indices[-1], 
                                    spike_indices[idx+1] if idx != len(spike_indices)-1 else global_max_time) 
                                    for idx in top_3_indices]

    fig, ax = plt.subplots(figsize=(15, len(neurons) * 0.5))
    neuron_ids = list(neurons.keys())
    for i, neuron_id in enumerate(neuron_ids):
        for j, (start, end) in enumerate(silent_periods[neuron_id][:3]):
            ax.fill_betweenx([i - 0.4, i + 0.4], start, end, color=plt.cm.viridis(j / 3), alpha=0.6,
                            label=f'Top {j+1}' if i == 0 else "")
    ax.set_yticks(np.arange(len(neuron_ids)))
    ax.set_yticklabels(neuron_ids)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron ID')
    ax.set_title('Top 3 Longest Silent Periods per Neuron')
    ax.set_xlim(0, global_max_time)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_neurons_silent_periods.png'))
    print(f'Saved at {os.path.join(save_dir, "all_neurons_silent_periods.png")}')
    plt.close()

def plot_spike_raster(neurons, save_dir, sample_rate, bin_size, cutoff_time=None):
    """Plot spike raster (full duration). If cutoff_time is set (same units as spike times), draw a vertical dashed line at last trial end."""
    neuron_spike_indices = {}
    global_max_time = 0
    firing_rates = {}
    for neuron_id, spike_times in neurons.items():
        spike_times = np.array(spike_times) / sample_rate * 1000
        spike_indices = np.round(spike_times).astype(int)
        global_max_time = max(global_max_time, int(np.max(spike_indices)) if len(spike_indices) > 0 else 0)
        neuron_spike_indices[neuron_id] = spike_indices
        # Calculate firing rate in Hz (spikes per second)
        # Use global_max_time in milliseconds, convert to seconds for Hz
        firing_rates[neuron_id] = len(spike_indices) / (global_max_time / 1000) if global_max_time > 0 else 0

    num_bins = int(np.ceil(global_max_time / bin_size)) + 1
    binned_spikes = {neuron_id: np.histogram(spike_indices, bins=num_bins, range=(0, global_max_time))[0] > 0 
                    for neuron_id, spike_indices in neuron_spike_indices.items()}

    fig, ax = plt.subplots(figsize=(15, len(neurons) * 0.5))
    neuron_ids = list(neurons.keys())
    for i, neuron_id in enumerate(neuron_ids):
        spike_bins = np.where(binned_spikes[neuron_id])[0]
        ax.plot(spike_bins * bin_size, [i] * len(spike_bins), 'k.', markersize=4, alpha=0.5)
    # Vertical dashed line at last TRIAL end (x in ms)
    if cutoff_time is not None:
        cutoff_ms = float(cutoff_time) / sample_rate * 1000
        ax.axvline(x=cutoff_ms, color='gray', linestyle='--', linewidth=1, label='Last TRIAL end')
    ax.set_yticks(np.arange(len(neuron_ids)))
    # Create labels with neuron ID and firing rate
    labels = [f'{neuron_id} ({firing_rates[neuron_id]:.2f} Hz)' for neuron_id in neuron_ids]
    ax.set_yticklabels(labels)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron ID')
    ax.set_title(f'Spike Raster Plot (Binned at {bin_size} ms)')
    ax.set_xlim(0, global_max_time)
    if cutoff_time is not None:
        ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spike_raster_binned.png'))
    print(f'Saved at {os.path.join(save_dir, "spike_raster_binned.png")}')
    plt.close()

# def compute_correlogram(x, y, max_lag_ms, bin_size, mode):
#     '''
#     x and y are the spike trains of the two neurons
#     max_lag_ms is the maximum lag in milliseconds
#     bin_size is the bin size in milliseconds
#     mode is either 'auto' or 'cross'
#     firing_rate_x and firing_rate_y are the firing rates of the two neurons
#     T is the total time of the spike trains
#     mean after normalizing by the expected coincidences is 1 as we did not remove the baseline firing rate
#     also return the std after normalizing by the expected coincidences
#     '''
#     # Convert max_lag from milliseconds to number of bins
#     max_lag_bins = int(max_lag_ms / bin_size)
    

#     x_centered = x 
#     y_centered = y 
    
#     # Compute correlation using numpy's correlate
#     corr = scipy.signal.correlate(x_centered, y_centered, mode='full')
#     # Find the center (zero lag) index
#     center = len(corr) // 2
    
#     # Extract only the desired lags
#     start_idx = center - max_lag_bins
#     end_idx = center + max_lag_bins + 1
#     corr_trimmed = corr[start_idx:end_idx]
    
#     # Zero out the center bin for autocorrelation
#     if mode == 'auto':
#         corr_trimmed[max_lag_bins] = 0
    
#     # Create lag values in milliseconds
#     lags = np.arange(-max_lag_bins, max_lag_bins + 1) * bin_size


#     mean = np.mean(corr_trimmed)  # Empirical mean (should be ~1)
#     std = np.std(corr_trimmed)   # Empirical std
#     z_scores = (corr_trimmed - mean) / std if std > 0 else np.zeros_like(corr_trimmed)


#     return lags, corr_trimmed, mean, std, z_scores


def compute_correlogram_normalized_cc_first(
    x, y, max_lag_ms, bin_size, mode,
    firing_rate_x, firing_rate_y, T, score_type='bump'):


    def bin_correlate(x, y, max_lag_ms, bin_size, T, mode):
        max_lag_bins = int(max_lag_ms / bin_size)
        x = np.pad(x, (0, T - len(x)), 'constant')
        y = np.pad(y, (0, T - len(y)), 'constant')

        corr = scipy.signal.correlate(y, x, mode='full')
        center = len(corr) // 2
        corr_trim = corr[center - max_lag_ms : center + max_lag_ms + 1]

        if mode == 'auto':
            corr_trim[max_lag_bins] -= np.sum(x)

        lags = np.arange(-max_lag_bins, max_lag_bins + 1) * bin_size
        binned = np.array([
            corr_trim[int(lags[i] + max_lag_ms):int(lags[i+1] + max_lag_ms)].sum()
            for i in range(2 * max_lag_bins)
        ])
        return (lags[:-1] + lags[1:]) / 2, binned

    def normalize(binned, fr_x, fr_y, T, bin_size):
        E = fr_x * fr_y * T * bin_size 
        return binned / E if E > 0 else binned.copy()

    def pct_empty(cc):
        return np.mean(cc < 1e-3)


    def bump_score(cc, baseline, max_lag):
        # max_lag is in milliseconds for one direction, so we need to double it for the full window width
        dev = np.maximum(cc - baseline, 0)           # 1. remove below baseline
        if dev.max() <= 0:
            return 0.0, np.zeros_like(cc)

        l0 = np.argmax(dev)                           # 2. center = biggest bin

        W = max_lag/2                               # full window width
        if W == 0:
            return dev[l0], np.zeros_like(cc)

        # 3. sigmoid curves (tanh stretched to go -1→1 over W/2 on each side)
        left_center  = l0 - W/2
        right_center = l0 + W/2

        l = np.arange(len(cc))
        s = np.full_like(cc, -1.0, dtype=float)

        left_mask  = (l >= l0 - W) & (l <= l0)
        right_mask = (l >  l0)     & (l <= l0 + W)

        s[left_mask]  = np.tanh(7 * (l[left_mask]  - left_center)  / W)
        s[right_mask] = np.tanh(7 * (right_center - l[right_mask]) / W)

        # 4. score = sum( dev(l) * s(l) )
        total = np.sum(dev * s) * (1 - pct_empty(cc))**2

        scores = dev * s   # optional: per-bin contribution
        # print(scores)
        return total, scores


    def multi_score(lags, cc):
        if len(cc) < 8:
            return 0.0
        samples = np.repeat(lags, np.round(cc * 10).astype(int))
        if len(samples) < 10:
            return 0.0
        dip, _ = diptest(samples)
        dev = np.abs(cc - cc.mean())
        p, props = find_peaks(dev, prominence=0.005 * cc.mean(), distance=2)
        max_prom = props['prominences'].max() / cc.mean() if len(p) else 0
        return dip * (1 + max_prom * 3)
    def snr_excess_area(cc_norm, mean_cc, std_cc):
        if mean_cc <= 0:
            return 0.0
        dev = cc_norm - mean_cc
        peaks, props = find_peaks(dev, prominence=0.01 * mean_cc)
        troughs, tprops = find_peaks(-dev, prominence=0.01 * mean_cc)
        
        peak_area = np.sum(props['prominences']) if len(peaks) else 0
        trough_area = np.sum(tprops['prominences']) if len(troughs) else 0
        
        norm_factor = std_cc if std_cc > 0 else 1
        return (peak_area + trough_area) / norm_factor  # Captures modes + deviations


    if T is None:
        T = max((x.max() if len(x) else 0), (y.max() if len(y) else 0)) + 1

    lags, binned = bin_correlate(x, y, max_lag_ms, bin_size, T, mode)
    cc_norm = normalize(binned, firing_rate_x, firing_rate_y, T, bin_size)

    mean_cc = cc_norm.mean()
    std_cc  = cc_norm.std()

    z = (cc_norm - mean_cc) / mean_cc if mean_cc > 0 else np.zeros_like(cc_norm)
    bs, bump_arr = bump_score(cc_norm, mean_cc, max_lag_ms)

    ms = multi_score(lags, cc_norm)

    above = np.maximum(cc_norm - mean_cc, 0)
    # signal to noise ratio excess area
    snr_excess_area = np.sum(above) / (np.std(cc_norm) + 1e-6)
    # snr excess area with penalty for extremely small bins
    pe = np.mean(cc_norm < 1e-3)
    snr_excess_area_with_penalty = snr_excess_area * (1 - pe)**2

    total = ms * (1 - pe)**2 * (1 + bs)

    if score_type == 'bump':
        return lags, cc_norm, mean_cc, std_cc, bump_arr, bs
    elif score_type == 'z_score':
        return lags, cc_norm, mean_cc, std_cc, z, total
    elif score_type == 'snr_excess_area':
        return lags, cc_norm, mean_cc, std_cc, snr_excess_area_with_penalty,snr_excess_area_with_penalty


def compute_correlogram_normalized(x, y, max_lag_ms, bin_size, mode, score_type='bump'):
    '''
    x and y are the spike trains of the two neurons
    max_lag_ms is the maximum lag in milliseconds
    bin_size is the bin size in milliseconds
    mode is either 'auto' or 'cross'
    firing_rate_x, firing_rate_y, T, edge_mean, score_type are included for compatibility
    Returns correlogram with normalization and scoring to match compute_correlogram_normalized output
    '''
    # Resample and binarize
    n_original = len(x)
    resampling_factor = int(bin_size / 1)  # Assuming original data is in 1ms bins
    x_resampled = np.array([np.sum(x[i:i+resampling_factor]) for i in range(0, n_original, resampling_factor)])
    y_resampled = np.array([np.sum(y[i:i+resampling_factor]) for i in range(0, n_original, resampling_factor)])
    x_resampled = (x_resampled > 0).astype(float)
    y_resampled = (y_resampled > 0).astype(float)
    
    # Compute correlation
    max_lag_bins = int(max_lag_ms / bin_size)
    corr = scipy.signal.correlate(y_resampled, x_resampled, mode='full')
    center = len(corr) // 2
    start_idx = center - max_lag_bins
    end_idx = center + max_lag_bins + 1
    corr_trimmed = corr[start_idx:end_idx]
    
    # Zero out zero-lag bin for autocorrelation
    if mode == 'auto':
        corr_trimmed[max_lag_bins] = 0
    else:
        print(f"Warning: mode is {mode}")
    
    # Create lag values in milliseconds
    lags = np.arange(-max_lag_bins, max_lag_bins + 1) * bin_size
    
    # Normalize by number of bins (simple normalization to produce corr_normalized)
    corr_normalized = corr_trimmed 

    
    # Compute mean and std
    mean_normalized = np.mean(corr_normalized)
    std_normalized = np.std(corr_normalized)
    
    # Compute scores
    if score_type == 'z_score':
        scores = (corr_normalized - mean_normalized) / std_normalized if std_normalized > 0 else np.zeros_like(corr_normalized)
        total_score = np.sum(np.abs(scores))
    else:  # score_type == 'bump'
        baseline = mean_normalized
        dev = np.abs(corr_normalized - baseline)
        from scipy.signal import find_peaks
        peaks, props = find_peaks(dev, prominence=0.01 * baseline)
        scores = np.zeros_like(corr_normalized)
        scores[peaks] = props['prominences'] / baseline
        total_score = np.sum(scores)
        # Add term: for each bump bin, cross_corr * (1 - percentage of empty bins); empty = literally 0
        pct_empty = np.mean(corr_normalized == 0)
        if len(peaks) > 0:
            total_score += np.sum(corr_normalized[peaks] * (1 - pct_empty))
    
    return lags, corr_normalized, mean_normalized, std_normalized, scores, total_score


def load_neurons(pkl_path, length_of_spiketrain=None, neuron_list=None):
    """Load spike train data from a .pkl file.

    Handles two formats:
    1. Simple: dict mapping neuron_id -> array of spike times (or binary spike train).
    2. Nested: dict with metadata and a 'neurons' key containing a list of dicts
       with 'name' and 'timestamps' (e.g. from NeuroSuite/neuroscope-style exports).

    Args:
        pkl_path: path to the .pkl file.
        length_of_spiketrain: optional cutoff; only keep spike times < this value.
        neuron_list: optional list of neuron names; if provided, only keep neurons whose name is in this list.

    Returns:
        neurons: dict of neuron_id -> spike times (full duration; caller applies cutoff if needed).
        rec_info: dict with 'tend' (total recording duration) and 'last_trial_end' (last TRIAL end time)
                  for nested format; both None for simple format.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    rec_info = {'tend': None, 'last_trial_end': None}

    # Nested format: {'neurons': [{'name': ..., 'timestamps': [...]}, ...], 'tend': ..., 'intervals': ...}
    if isinstance(data, dict) and 'neurons' in data:
        neurons_list = data['neurons']
        if not isinstance(neurons_list, (list, tuple)):
            raise ValueError("Expected 'neurons' to be a list of dicts with 'name' and 'timestamps'")
        neurons = {}
        for item in neurons_list:
            if isinstance(item, dict) and 'name' in item and 'timestamps' in item:
                name = item['name']
                if neuron_list is not None and name not in neuron_list:
                    continue
                ts = np.asarray(item['timestamps'], dtype=float)
                neurons[name] = ts
            else:
                raise ValueError(f"Each entry in 'neurons' must have 'name' and 'timestamps'; got keys: {item.keys() if isinstance(item, dict) else type(item)}")
        if 'tend' in data:
            rec_info['tend'] = float(data['tend'])
        if 'events' in data:
            trials = data['events'][-1] if  data['events'][-1]['name'] == 'TRIAL' else None
            if trials is not None:
                rec_info['last_trial_end'] = trials['timestamps'][-1]

    else:
        # Simple format: dict of neuron_id -> spike array
        neurons = {k: np.asarray(v, dtype=float) for k, v in data.items()}
        if neuron_list is not None:
            neurons = {k: v for k, v in neurons.items() if k in neuron_list}

    # Detect binary spike train (many zeros) and convert to spike indices
    if neurons:
        first_key = next(iter(neurons.keys()))
        arr = np.asarray(neurons[first_key])
        if arr.size > 100 and np.sum(arr == 0) > 100:
            neurons = {k: np.where(np.asarray(v))[0] for k, v in neurons.items()}

    # Optional caller-provided cutoff (e.g. for simple format); do not apply rec_info cutoff here
    if length_of_spiketrain is not None:
        neurons = {k: v[v < length_of_spiketrain] for k, v in neurons.items()}

    return neurons, rec_info



def plot_neuron_correlation_matrices(neurons, save_dir, sample_rate=1, edge_mean=True, configs=None, make_plots=True):
    if configs is None:
        configs = [(80, 1000, "Broad"), (40, 500, "Medium"), (10, 100, "semiFine"),(5, 100, "Fine")]
    neuron_ids = list(neurons.keys())
    
    for bin_size, max_lag, resolution in configs:
        neuron_spike_trains, global_max_time, firing_rate, global_max_times = {}, 0, {}, {}
        for neuron_id, spike_times in neurons.items():
            spike_times = np.array(spike_times) / sample_rate * 1000
            spike_indices = np.round(spike_times).astype(int)
            global_max_time = max(global_max_time, int(np.max(spike_indices)) if len(spike_indices) > 0 else 0)
            global_max_times[neuron_id] = global_max_time
            num_bins = global_max_time
            neuron_spike_trains[neuron_id] = np.histogram(spike_indices, bins=num_bins, range=(0, global_max_time))[0] > 0
            firing_rate[neuron_id] = np.sum(neuron_spike_trains[neuron_id]) / global_max_time 

        autocorrs = {n: compute_correlogram_normalized(neuron_spike_trains[n].astype(float), neuron_spike_trains[n].astype(float), max_lag, bin_size, 'auto') 
                    for n in neuron_ids}
        crosscorrs = {(pre, post): compute_correlogram_normalized_cc_first(neuron_spike_trains[pre].astype(float), neuron_spike_trains[post].astype(float), max_lag, bin_size, 'cross', firing_rate[pre], firing_rate[post], max(global_max_times[pre], global_max_times[post]))
                     for pre in neuron_ids for post in neuron_ids if pre != post}
        if make_plots:
            fig_size = min(24, 1.2 * (len(neuron_ids) + 1))
            fig = plt.figure(figsize=(fig_size, fig_size))
            gs = fig.add_gridspec(len(neuron_ids) + 1, len(neuron_ids) + 1, hspace=0.15, wspace=0.15)
            
            for i, pre_id in enumerate(neuron_ids):
                for j, post_id in enumerate(neuron_ids):
                    ax = fig.add_subplot(gs[i + 1, j + 1])
                    if pre_id != post_id:
                        lags, corr, mean_corr, std_corr, z_score, total_bump_score = crosscorrs[(pre_id, post_id)]
                        # Same plot style as pipeline rank fig: stair fill 0→line, red, mean line, gray ±std band
                        n = len(corr)
                        if len(lags) > n:
                            x_stair = np.concatenate([np.repeat(lags[:n], 2), [lags[n]]])
                            y_stair = np.concatenate([np.repeat(corr, 2), [corr[-1]]])
                        else:
                            x_stair = np.repeat(lags, 2)
                            y_stair = np.repeat(corr, 2)
                        ax.fill_between(x_stair, 0, y_stair, 
                            step="pre",          # ← this is the key
                            color='red', alpha=0.8, 
                            edgecolor='none', linewidth=0)
                        ax.axhline(y=mean_corr, color='black', linestyle='--', linewidth=0.5, alpha=0.8)
                        ax.fill_between(
                            [lags[0], lags[-1]],
                            [mean_corr - std_corr, mean_corr - std_corr],
                            [mean_corr + std_corr, mean_corr + std_corr],
                            color='gray', alpha=0.2, edgecolor='none', linewidth=0)
                    ax.set_xlim(-max_lag, max_lag)
                    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.3)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                    if i < len(neuron_ids) - 1:
                        ax.set_xticklabels([])
                    else:
                        ax.set_xlabel('ms', fontsize=6)
                    ax.tick_params(labelsize=5)
                    if i == 0:
                        # Truncate to max 5 letters for title, removing "neuron" prefix
                        post_id_short = post_id.replace("neuron", "")[:5]
                        ax.set_title(post_id_short, fontsize=6, pad=2)
                    if j == 0:
                        # Truncate to max 5 letters for ylabel, removing "neuron" prefix
                        pre_id_short = pre_id.replace("neuron", "")[:5]
                        ax.set_ylabel(pre_id_short, fontsize=6)
                    ax.set_box_aspect(1)
            
            for idx, neuron_id in enumerate(neuron_ids):
                lags, corr, mean_corr, std_corr, z_score, total_bump_score = autocorrs[neuron_id]
                y_range = np.max(corr) - np.min(corr)
                ax_top = fig.add_subplot(gs[0, idx + 1])
                # Fill under the .step plot with color
                ax_top.fill_between(lags, 0, corr, color='blue', alpha=1, step='mid', linewidth=0)
                ax_top.step(lags, corr, where='mid', color='b', linewidth=0)
                ax_top.set_xlim(-max_lag, max_lag)
                ax_top.set_ylim(np.min(corr) - 0.1 * y_range, np.max(corr) + 0.1 * y_range)
                ax_top.grid(True, linestyle=':', linewidth=0.5, alpha=0.3)
                ax_top.tick_params(labelsize=5)
                ax_top.spines['top'].set_visible(False)
                ax_top.spines['right'].set_visible(False)
                ax_top.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                # Truncate to max 5 letters for title, removing "neuron" prefix
                neuron_id_short = neuron_id.replace("neuron", "")[:5]
                ax_top.set_title(neuron_id_short, fontsize=6, pad=2)
                ax_top.set_box_aspect(1)
                
                ax_left = fig.add_subplot(gs[idx + 1, 0])
                # Fill under the .step plot with color
                ax_left.fill_between(lags, 0, corr, color='blue', alpha=1, step='mid', linewidth=0)
                ax_left.step(lags, corr, where='mid', color='b', linewidth=0)
                ax_left.set_xlim(-max_lag, max_lag)
                ax_left.set_ylim(np.min(corr) - 0.1 * y_range, np.max(corr) + 0.1 * y_range)
                ax_left.grid(True, linestyle=':', linewidth=0.5, alpha=0.3)
                ax_left.tick_params(labelsize=5)
                ax_left.spines['top'].set_visible(False)
                ax_left.spines['right'].set_visible(False)
                ax_left.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                # Truncate to max 5 letters for ylabel, removing "neuron" prefix
                neuron_id_short = neuron_id.replace("neuron", "")[:5]
                ax_left.set_ylabel(neuron_id_short, fontsize=6)
                ax_left.set_box_aspect(1)
            
            ax_corner = fig.add_subplot(gs[0, 0])
            ax_corner.axis('off')
            ax_corner.set_box_aspect(1)
            
            # Add global note about input-output relationship
            ax_corner.text(0.5, 0.5, 'Output TopRow→\n↓\nLeftColInput', 
                          transform=ax_corner.transAxes,
                          ha='center', va='center',
                          fontsize=8, fontweight='bold',
                          bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
            
            plt.suptitle(f'Cross-correlation Matrix ({resolution} Resolution: ±{max_lag}ms, {bin_size}ms bins)', fontsize=10, y=0.98)
            plt.tight_layout()
            os.makedirs(save_dir, exist_ok=True)
            filedir = os.path.join(save_dir, f'correlation_matrix_edge_mean_{edge_mean}_{resolution.lower()}.png')
            plt.savefig(filedir, dpi=300, bbox_inches='tight')
            print(f'Saved {resolution} resolution matrix at {filedir}')
            plt.show()
            plt.close()
        # with open(os.path.join(save_dir, f'crosscorrs_edge_mean_{edge_mean}_{resolution.lower()}.pkl'), 'wb') as f:
        #     pickle.dump(crosscorrs, f)
        # with open(os.path.join(save_dir, f'autocorrs_edge_mean_{edge_mean}_{resolution.lower()}.pkl'), 'wb') as f:
        #     pickle.dump(autocorrs, f)
        # print(f'Saved {resolution} resolution crosscorrs and autocorrs at {save_dir}')
    return configs



