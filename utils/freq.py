import numpy as np
from scipy.signal import find_peaks


def band_peak_power(psd_ch, freqs, fmin, fmax):
    """Power at the highest *local* peak in [fmin, fmax].

    Uses scipy.signal.find_peaks to detect interior maxima, so the
    left-edge 2 Hz point (elevated by 1/f) cannot win unless it is an
    actual local peak.  Falls back to the band maximum when no interior
    peak exists (monotone spectrum).
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    y = psd_ch[mask]
    peaks, _ = find_peaks(y)  # only interior local maxima
    if len(peaks):
        return y[peaks[np.argmax(y[peaks])]]
    return y.max()  # no interior peak – return max-in-band


def run_freq_analysis(eeg_data, params):
    """
    Compute per-channel Welch PSD, detect outlier channels via MAD, and
    identify the best channel (highest in-band peak) per participant.

    Parameters
    ----------
    eeg_data : dict
        Output of Things_Importer.get_eeg_data().
    params : dict
        FMIN_PSD, FMAX_PSD, FMIN_BAND, FMAX_BAND, N_FFT_OSC,
        OUTLIER_THRESH_MAD, and optionally "segment" (default "cued_replay")

    Returns
    -------
    freq_data : dict
        psd_data       - dict  p → (n_ch, n_freqs) or None
        psd_freqs      - (n_freqs,)
        avg_power      - (n_subs, n_ch)
        peak_power     - (n_subs, n_ch)
        ch_outlier     - (n_subs, n_ch) bool
        max_ch_idx_peak - (n_subs,)
        max_ch_peak_vals - (n_subs,)
    """
    FMIN_PSD = params["FMIN_PSD"]
    FMAX_PSD = params["FMAX_PSD"]
    FMIN_BAND = params["FMIN_BAND"]
    FMAX_BAND = params["FMAX_BAND"]
    N_FFT_OSC = params["N_FFT_OSC"]
    OUTLIER_THRESH_MAD = params["OUTLIER_THRESH_MAD"]
    segment = params.get("segment", "cued_replay")

    epochs_list = eeg_data[segment]["epochs"]
    n_subs = len(epochs_list)

    psd_data = {}
    psd_freqs = None

    for p, epochs in enumerate(epochs_list):
        if epochs is None:
            psd_data[p] = None
            continue
        psd_obj = epochs.compute_psd(
            method="welch",
            fmin=FMIN_PSD,
            fmax=FMAX_PSD,
            n_fft=N_FFT_OSC,
            n_overlap=N_FFT_OSC // 2,
            verbose=False,
        )
        if psd_freqs is None:
            psd_freqs = psd_obj.freqs
        psd_data[p] = psd_obj.get_data().mean(axis=0)

    ch_names = eeg_data["resting"]["ch_names"]
    n_ch = len(ch_names)
    band_mask = (psd_freqs >= FMIN_BAND) & (psd_freqs <= FMAX_BAND)

    _band_pwr = np.full((n_subs, n_ch), np.nan)
    for p in range(n_subs):
        if psd_data.get(p) is not None:
            _band_pwr[p] = psd_data[p][:, band_mask].mean(axis=-1)

    ch_outlier = np.zeros((n_subs, n_ch), dtype=bool)
    for p in range(n_subs):
        row = _band_pwr[p]
        if np.all(np.isnan(row)):
            continue
        valid = row[~np.isnan(row)]
        med = np.median(valid)
        mad = np.median(np.abs(valid - med))
        if mad == 0:
            continue
        ch_outlier[p] = row > med + OUTLIER_THRESH_MAD * mad

    n_flagged = int(ch_outlier.sum())
    n_valid_subs = (~np.all(np.isnan(_band_pwr), axis=1)).sum()
    print(
        f"Outlier channels flagged: {n_flagged} "
        f"({n_flagged / max(1, n_valid_subs * n_ch) * 100:.1f}% "
        f"of all valid participant × channel entries)"
    )

    avg_power = np.full((n_subs, n_ch), np.nan)
    peak_power = np.full((n_subs, n_ch), np.nan)

    for p in range(n_subs):
        if psd_data.get(p) is None:
            continue
        for ch in range(n_ch):
            if ch_outlier[p, ch]:
                continue
            avg_power[p, ch] = psd_data[p][ch, band_mask].mean()
            peak_power[p, ch] = band_peak_power(psd_data[p][ch], psd_freqs, FMIN_BAND, FMAX_BAND)

    max_ch_idx_peak = np.nanargmax(peak_power, axis=1)
    max_ch_peak_vals = peak_power[np.arange(n_subs), max_ch_idx_peak]

    return {
        "psd_data": psd_data,
        "psd_freqs": psd_freqs,
        "avg_power": avg_power,
        "peak_power": peak_power,
        "ch_outlier": ch_outlier,
        "max_ch_idx_peak": max_ch_idx_peak,
        "max_ch_peak_vals": max_ch_peak_vals,
    }
