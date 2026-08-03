from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks as _find_peaks
from scipy.stats import linregress, spearmanr

matplotlib.use("qtagg")

from utils.paths import GRAPHICS_DIR
from utils.plots import plot_cohens_kappa, plot_data_quant, plot_sequenceness

# Data Quantification: Trial Numbers and Resting Times Old Infants

plot_data_quant(
    behavioral_data_old["localizer"]["n_trials"],
    behavioral_data_old["resting"]["s_length"],
    ["Apple", "Chair", "Face"],
    behavioral_data_old["info_names"],
    colors=("#1f77b4ff", "#ff7f0eff", "#2ca02cff"),
    save=os.path.join(GRAPHICS_DIR, "Results", f"Things_DataQuanitification_old.svg"),
)
plt.show(block=False)


plot_data_quant(
    behavioral_data_young["localizer"]["n_trials"],
    behavioral_data_young["resting"]["s_length"],
    ["Apple", "Chair", "Face"],
    behavioral_data_young["info_names"],
    colors=("#1f77b4ff", "#ff7f0eff", "#2ca02cff"),
    save=os.path.join(GRAPHICS_DIR, "Results", f"Things_DataQuanitification_young.svg"),
)
plt.show(block=False)

#######################################
"""
Localizer: Time-resolved decoding Single Participant

"""
# Older infants

fig, ax = plt.subplots(
    nrows=4,
    ncols=5,
    subplot_kw={},
    gridspec_kw={
        "height_ratios": [1] * 4,
        "width_ratios": [1] * 5,
        "hspace": 0.3,
        "wspace": 0.1,
    },
    figsize=(15, 9),
)
for i, a in enumerate(ax.flat):
    a.axhline(0.5, color="k", linestyle="--", label="Chance Level")
    a.axvline(0.0, color="k", linestyle="-")
    a.plot(
        eeg_data_old["localizer"]["times"],
        classifier_data_old["performance"][i, :, :].mean(0),
        label="Class. Perf.",
    )
    a.fill_between(
        eeg_data_old["localizer"]["times"],
        classifier_data_old["performance"][i, :, :].mean(0)
        - classifier_data_old["performance"][i, :, :].std(0),
        classifier_data_old["performance"][i, :, :].mean(0)
        + classifier_data_old["performance"][i, :, :].std(0),
        alpha=0.2,
    )
    a.set_title(f"Participant {i + 1} - {participant_info_old['additional_data']['info_names'][i]}")
    a.set_ylim([0.3, 0.8])
    a.set_xlim([-0.2, 0.8])
    a.set_xticks(np.arange(-0.2, 0.81, 0.2))
    a.set_yticks(np.arange(0.3, 0.81, 0.1))
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Localizer_Individual_old.png"), dpi=300)
[a.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[]) for a in ax.flat]
fig.savefig(Path(GRAPHICS_DIR, "Results", "Localizer_Individual_old.svg"), dpi=300)

# Younger infants

fig, ax = plt.subplots(
    nrows=5,
    ncols=5,
    subplot_kw={},
    gridspec_kw={
        "height_ratios": [1] * 5,
        "width_ratios": [1] * 5,
        "hspace": 0.3,
        "wspace": 0.1,
    },
    figsize=(15, 11.25),
)
for i, a in enumerate(ax.flat):
    if i < n_subs_young:
        a.axhline(0.5, color="k", linestyle="--", label="Chance Level")
        a.axvline(0.0, color="k", linestyle="-")
        a.plot(
            eeg_data_young["localizer"]["times"],
            classifier_data_young["performance"][i, :, :].mean(0),
            label="Class. Perf.",
        )
        a.fill_between(
            eeg_data_young["localizer"]["times"],
            classifier_data_young["performance"][i, :, :].mean(0)
            - classifier_data_young["performance"][i, :, :].std(0),
            classifier_data_young["performance"][i, :, :].mean(0)
            + classifier_data_young["performance"][i, :, :].std(0),
            alpha=0.2,
        )
        a.set_title(
            f"Participant {i + 1} - {participant_info_young['additional_data']['info_names'][i]}"
        )
        a.set_ylim([0.3, 0.8])
        a.set_xlim([-0.2, 0.8])
        a.set_xticks(np.arange(-0.2, 0.81, 0.2))
        a.set_yticks(np.arange(0.3, 0.81, 0.1))
    else:
        a.set_axis_off()
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Localizer_Individual_young.png"), dpi=300)
[a.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[]) for a in ax.flat]
fig.savefig(Path(GRAPHICS_DIR, "Results", "Localizer_Individual_young.svg"), dpi=300)


#######################################
"""
Cued Replay: Empirical Transition Matrices

"""

vmin = -0.2
vmax = 0.8

emp_tm_old = np.transpose(xcorr_data_old["cued_replay"]["empirical_tm"].mean(0), (2, 0, 1))

# Visualizations
fig, axes = plt.subplots(5, 10, figsize=(12, 7))

for i, ax in enumerate(axes.flat):
    ax.axis("off")
    if emp_tm_old.shape[0] > i:
        im = ax.imshow(
            emp_tm_old[i, :, :],
            vmin=vmin,
            vmax=vmax,
            cmap="Blues",
            aspect="equal",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{(i + 1) * 10} ms", {"fontsize": 10})
cbar_ax = fig.add_axes([0.96, 0.35, 0.006, 0.25])
cbar = fig.colorbar(im, cax=cbar_ax)
plt.tight_layout(rect=[0, 0, 0.94, 1], w_pad=0.6, pad=0.02)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "CuedReplay_EmpTM_old.png"), dpi=300)
cbar.set_ticklabels("")
fig.savefig(Path(GRAPHICS_DIR, "Results", "CuedReplay_EmpTM_old.svg"), dpi=300)


emp_tm_young = np.transpose(xcorr_data_young["cued_replay"]["empirical_tm"].mean(0), (2, 1, 0))

# Visualizations
fig, axes = plt.subplots(5, 10, figsize=(12, 7))

for i, ax in enumerate(axes.flat):
    ax.axis("off")
    if emp_tm_young.shape[0] > i:
        im = ax.imshow(
            emp_tm_young[i, :, :],
            vmin=vmin,
            vmax=vmax,
            cmap="Blues",
            aspect="equal",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{(i + 1) * 10} ms", {"fontsize": 10})
cbar_ax = fig.add_axes([0.96, 0.35, 0.006, 0.25])
cbar = fig.colorbar(im, cax=cbar_ax)
plt.tight_layout(rect=[0, 0, 0.94, 1], w_pad=0.6, pad=0.02)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "CuedReplay_EmpTM_young.png"), dpi=300)
cbar.set_ticklabels("")
fig.savefig(Path(GRAPHICS_DIR, "Results", "CuedReplay_EmpTM_young.svg"), dpi=300)


#######################################
"""
Cued Replay: Theta-Frequency Correlation

"""

r_old, p_val_old = spearmanr(seq_net_effect_old, freq_data_old["max_ch_peak_vals"] * 1e12)

# Trend line (OLS)
slope_old, intercept_old, *_ = linregress(
    freq_data_old["max_ch_peak_vals"] * 1e12, seq_net_effect_old
)
x_line_old = np.linspace(
    freq_data_old["max_ch_peak_vals"].min() * 1e12,
    freq_data_old["max_ch_peak_vals"].max() * 1e12,
    200,
)

fig, ax = plt.subplots(figsize=(6, 5))
ax.axhline(0, color="dimgrey", linewidth=2.0, linestyle="--")
ax.plot(
    x_line_old,
    slope_old * x_line_old + intercept_old,
    color="black",
    linewidth=2.5,
    linestyle="--",
    zorder=2,
)
ax.scatter(
    freq_data_old["max_ch_peak_vals"] * 1e12,
    seq_net_effect_old,
    color="steelblue",
    s=75,
    alpha=1.0,
    edgecolors="white",
    linewidths=0.5,
)
ax.set_xlabel("Individual Peak Theta Power (µV²/Hz)", fontsize=12)
ax.set_ylabel("Avg net sequenceness effect", fontsize=12)
ax.tick_params(axis="both", which="major", length=8)
plt.tight_layout(pad=0.01)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "CuedReplay_ThetaCorrelation.png"), dpi=300)
ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
plt.tight_layout(pad=0.01)
fig.savefig(Path(GRAPHICS_DIR, "Results", "CuedReplay_ThetaCorrelation.svg"), dpi=300)


# PSD – best channel per participant (old group, all participants overlaid)

# Collect best-channel PSD (V²/Hz) for each valid participant
psds_old = []
for p in range(n_subs_old):
    if freq_data_old["psd_data"].get(p) is None:
        continue
    psds_old.append(freq_data_old["psd_data"][p][freq_data_old["max_ch_idx_peak"][p]])

n_valid_old = len(psds_old)
cmap_psd = plt.cm.get_cmap("viridis", n_valid_old)

_band_mask = (freq_data_old["psd_freqs"] >= FMIN_BAND) & (freq_data_old["psd_freqs"] <= FMAX_BAND)
_freqs_band = freq_data_old["psd_freqs"][_band_mask]

fig, ax = plt.subplots(figsize=(6, 4))

for i, psd in enumerate(psds_old):
    color = cmap_psd(i / max(n_valid_old - 1, 1))
    psd_uv = psd * 1e12  # V²/Hz → µV²/Hz
    ax.plot(freq_data_old["psd_freqs"], psd_uv, color=color, lw=1.0, alpha=0.75)

    # marker at dominant local peak in the band of interest
    y_band = psd_uv[_band_mask]
    peaks, _ = _find_peaks(y_band)
    best = peaks[np.argmax(y_band[peaks])] if len(peaks) else np.argmax(y_band)
    ax.plot(
        _freqs_band[best],
        y_band[best],
        "o",
        color=color,
        ms=5,
        zorder=5,
        markeredgecolor="white",
        markeredgewidth=0.5,
    )

# Group mean
if psds_old:
    ax.plot(
        freq_data_old["psd_freqs"],
        np.stack(psds_old).mean(0) * 1e12,
        color="k",
        lw=2.5,
        label=f"Mean (n={n_valid_old})",
        zorder=6,
    )


ax.axvspan(FMIN_BAND, FMAX_BAND, alpha=0.10, color="gold", zorder=0)
ax.set_xlim(2, 16)
ax.set_ylim(0, 500)
ax.set_xlabel("Frequency (Hz)", fontsize=12)
ax.set_ylabel("Power (µV²/Hz)", fontsize=12)
ax.legend(fontsize=10, frameon=False)
ax.tick_params(axis="both", which="major", length=6)
plt.tight_layout(pad=0.3)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "CuedReplay_PSD_BestChannel.png"), dpi=300)
ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
plt.tight_layout(pad=0.01)
fig.savefig(Path(GRAPHICS_DIR, "Results", "CuedReplay_PSD_BestChannel.svg"), dpi=300)


#######################################
"""
Sequence Learning: Sequenceness

"""

fig, ax = plt.subplots(figsize=(7, 3))
plot_sequenceness(
    xcorr_data_old["seq_learn"]["seq_net"],
    lags_ms,
    xcorr_data_old["seq_learn"]["perm_net"],
    plot_dist_perm=None,
    plot_max_perm=False,
    plot_subs=False,
    ax=ax,
)
ax.set_xlim([0, MAX_LAG * 10])
ax.set_ylim([-0.06, 0.06])
ax.get_lines()[0].set_linewidth(2.0)
ax.get_lines()[1].set_linewidth(2.5)
thresh_t = xcorr_data_old["seq_learn"]["threshold_data"]
ax.axhline(thresh_t, color="dimgrey", linestyle="--", linewidth=2.0)
ax.axhline(-thresh_t, color="dimgrey", linestyle="--", linewidth=2.0)
ax.set_xlabel("time lag (ms)", fontsize=12)
ax.set_ylabel("sequenceness (fwd − bwd)", fontsize=12)
ax.tick_params(axis="both", which="major", length=8)
ax.tick_params(axis="both", which="minor", length=4)
ax.legend().remove()
plt.tight_layout(pad=0.02)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Sequence_Learning_Sequenceness_old.png"), dpi=300)
ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
plt.tight_layout(pad=0.02)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Sequence_Learning_Sequenceness_old.svg"), dpi=300)

print(
    f"Sequence presentation - Old: Permutation t-test against 0\n"
    f"t_max = {xcorr_data_old['seq_learn']['stat_info']['max_real']}, "
    f"p = {xcorr_data_old['seq_learn']['stat_info']['pval']}"
)


fig, ax = plt.subplots(figsize=(7, 3))
plot_sequenceness(
    xcorr_data_young["seq_learn"]["seq_net"],
    lags_ms,
    xcorr_data_young["seq_learn"]["perm_net"],
    plot_dist_perm=None,
    plot_max_perm=False,
    plot_subs=False,
    ax=ax,
)
ax.set_xlim([0, MAX_LAG * 10])
ax.set_ylim([-0.06, 0.06])
ax.get_lines()[0].set_linewidth(2.0)
ax.get_lines()[1].set_linewidth(2.5)
thresh_t = xcorr_data_young["seq_learn"]["threshold_data"]
ax.axhline(thresh_t, color="dimgrey", linestyle="--", linewidth=2.0)
ax.axhline(-thresh_t, color="dimgrey", linestyle="--", linewidth=2.0)
ax.set_xlabel("time lag (ms)", fontsize=12)
ax.set_ylabel("sequenceness (fwd − bwd)", fontsize=12)
ax.tick_params(axis="both", which="major", length=8)
ax.tick_params(axis="both", which="minor", length=4)
ax.legend().remove()
plt.tight_layout(pad=0.02)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Sequence_Learning_Sequenceness_young.png"), dpi=300)
ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
plt.tight_layout(pad=0.02)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Sequence_Learning_Sequenceness_young.svg"), dpi=300)

print(
    f"Sequence presentation - Young: Permutation t-test against 0\n"
    f"t_max = {xcorr_data_young['seq_learn']['stat_info']['max_real']}, "
    f"p = {xcorr_data_young['seq_learn']['stat_info']['pval']}"
)


#######################################
"""
Preresting: Sequenceness

"""

fig, ax = plt.subplots(figsize=(7, 3))
plot_sequenceness(
    xcorr_data_old["preresting"]["seq_net"],
    lags_ms,
    xcorr_data_old["preresting"]["perm_net"],
    plot_dist_perm=None,
    plot_max_perm=False,
    plot_subs=False,
    ax=ax,
)
ax.set_xlim([0, MAX_LAG * 10])
ax.set_ylim([-0.06, 0.06])
ax.get_lines()[0].set_linewidth(2.0)
ax.get_lines()[1].set_linewidth(2.5)
thresh_t = xcorr_data_old["seq_learn"]["threshold_data"]
ax.axhline(thresh_t, color="dimgrey", linestyle="--", linewidth=2.0)
ax.axhline(-thresh_t, color="dimgrey", linestyle="--", linewidth=2.0)
ax.set_xlabel("time lag (ms)", fontsize=12)
ax.set_ylabel("sequenceness (fwd − bwd)", fontsize=12)
ax.tick_params(axis="both", which="major", length=8)
ax.tick_params(axis="both", which="minor", length=4)
ax.legend().remove()
plt.tight_layout(pad=0.02)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Preresting_Sequenceness_old.png"), dpi=300)
ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
plt.tight_layout(pad=0.02)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Preresting_Sequenceness_old.svg"), dpi=300)

print(
    f"Preresting - Old: Permutation t-test against 0\n"
    f"t_max = {xcorr_data_old['preresting']['stat_info']['max_real']}, "
    f"p = {xcorr_data_old['preresting']['stat_info']['pval']}"
)


fig, ax = plt.subplots(figsize=(7, 3))
plot_sequenceness(
    xcorr_data_young["preresting"]["seq_net"],
    lags_ms,
    xcorr_data_young["preresting"]["perm_net"],
    plot_dist_perm=None,
    plot_max_perm=False,
    plot_subs=False,
    ax=ax,
)
ax.set_xlim([0, MAX_LAG * 10])
ax.set_ylim([-0.06, 0.06])
ax.get_lines()[0].set_linewidth(2.0)
ax.get_lines()[1].set_linewidth(2.5)
thresh_t = xcorr_data_young["seq_learn"]["threshold_data"]
ax.axhline(thresh_t, color="dimgrey", linestyle="--", linewidth=2.0)
ax.axhline(-thresh_t, color="dimgrey", linestyle="--", linewidth=2.0)
ax.set_xlabel("time lag (ms)", fontsize=12)
ax.set_ylabel("sequenceness (fwd − bwd)", fontsize=12)
ax.tick_params(axis="both", which="major", length=8)
ax.tick_params(axis="both", which="minor", length=4)
ax.legend().remove()
plt.tight_layout(pad=0.02)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Preresting_Sequenceness_young.png"), dpi=300)
ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
plt.tight_layout(pad=0.02)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Preresting_Sequenceness_young.svg"), dpi=300)

print(
    f"Preresting - Young: Permutation t-test against 0\n"
    f"t_max = {xcorr_data_young['preresting']['stat_info']['max_real']}, "
    f"p = {xcorr_data_young['preresting']['stat_info']['pval']}"
)


# Save trial info CSVs


def _save_trial_info_csv(behavioral_data, participant_info, save_path):
    if "seq_learn" not in behavioral_data:
        return
    loc_trialnum_sum = behavioral_data["localizer"]["n_trials"].sum(axis=1)
    seq_reps = behavioral_data["seq_learn"]["n_trials"].flatten()
    pd.DataFrame(
        {
            "Participant": participant_info["pilotnames_incl"],
            "Age_Months": participant_info["participant_included"]["Age_Months"].values,
            "Gender": participant_info["participant_included"]["Gender"].values,
            "Localizer_Trials": loc_trialnum_sum,
            "Localizer_Trials_Percent": loc_trialnum_sum / 540,
            "Sequence_Presentations": seq_reps,
            "Sequence_Presentations_Percent": seq_reps / 100,
        }
    ).to_csv(save_path, index=False)


_save_trial_info_csv(
    behavioral_data_old,
    participant_info_old,
    Path(ROOT, "Analysis", "scripts", "additional_data", "trial_info_old.csv"),
)
_save_trial_info_csv(
    behavioral_data_young,
    participant_info_young,
    Path(ROOT, "Analysis", "scripts", "additional_data", "trial_info_young.csv"),
)


# Plot Cohen's kappa over groups

group_cohkappa = np.concatenate(
    [
        participant_info_old["additional_data"]["localizer"]["cohkappa_scores"],
        participant_info_young["additional_data"]["localizer"]["cohkappa_scores"],
    ]
)

plot_cohens_kappa(group_cohkappa)
plt.show(block=False)

print(
    f"Cohen's kappa over groups: k = {group_cohkappa.mean():.3f}, SD = {group_cohkappa.std():.3f}"
)
