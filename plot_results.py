from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.stats import spearmanr, wilcoxon

matplotlib.use("qtagg")

from utils.paths import GRAPHICS_DIR
from utils.plots import (
    add_significance,
    plot_sequenceness,
    plot_sliding_classifier,
    plot_topo_class_pattern,
)

# ==============================================================================
# PARTICIPANT SUMMARY
# ==============================================================================


def _print_group_summary(label, behavioral_data, n_subs):
    ad = behavioral_data
    loc = ad["localizer"]
    cr = ad["cued_replay"]
    rst = ad["resting"]
    sl = ad["seq_learn"]
    pre = ad.get("preresting", {})

    print("═" * 62)
    print(f" {label}  (n={n_subs}, mean age {ad['mean_age']:.1f} mo)")
    print("─" * 62)
    print(
        f" Localizer:    {loc['n_trials'].sum(1).mean():.0f} ± "
        f"{loc['n_trials'].sum(1).std():.0f} trials  |  "
        f"rejection {loc['mean_trlrej'] * 100:.1f} ± {loc['std_trlrej'] * 100:.1f}%"
    )
    print(
        f" Cued Replay:  {cr['n_trials'].sum(1).mean():.0f} ± "
        f"{cr['n_trials'].sum(1).std():.0f} trials  |  "
        f"rejection {cr['mean_trlrej'] * 100:.1f} ± {cr['std_trlrej'] * 100:.1f}%"
    )
    print(
        f" Resting:      {rst['s_length'][:, 0].mean():.1f} ± {rst['s_length'][:, 0].std():.1f} s"
    )
    print(
        f" Seq. Learn:   {sl['n_trials'].sum(1).mean():.1f} ± "
        f"{sl['n_trials'].sum(1).std():.1f} presentations  |  "
        f"rejection {sl['mean_trlrej'] * 100:.1f} ± {sl['std_trlrej'] * 100:.1f}%"
    )
    if pre:
        print(
            f" Pre-Resting:  {pre['s_length'][:, 0].mean():.1f} ± "
            f"{pre['s_length'][:, 0].std():.1f} s"
        )
    print("═" * 62)


_print_group_summary("OLD GROUP ", behavioral_data_old, n_subs_old)
_print_group_summary("YOUNG GROUP", behavioral_data_young, n_subs_young)


#######################################
"""
Localizer: Time-resolved decoding and spatial patterns

"""

fig, ax = plt.subplots(figsize=(7, 3.5))
plot_sliding_classifier(
    classifier_data_old["performance"].mean(1),
    eeg_data_old["localizer"]["times"],
    chance=0.5,
    ax=ax,
)
ax.set_xlim([-0.2, 0.8])
ax.set_ylim([0.45, 0.70])
ax.set_yticks(np.arange(0.45, 0.71, 0.025))
ax.get_lines()[0].set_linewidth(2.0)
ax.get_lines()[1].set_linewidth(2.0)
ax.get_lines()[2].set_linewidth(2.5)
ax.legend().remove()
ax.tick_params(axis="both", length=8)
add_significance(
    ax=ax,
    stat_info=classifier_data_old["group_stats"]["cluster_dict"],
    times=eeg_data_old["localizer"]["times"],
    plot_type="linepoints",
    y_pos=0.475,
    color="k",
)
plt.tight_layout(pad=0.02)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Localizer_TimeCourse_old.png"), dpi=300)
ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
plt.tight_layout(pad=0.02)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Localizer_TimeCourse_old.svg"), dpi=300)


spatial_patterns_rmse = [
    np.transpose(np.std(sp, axis=0, keepdims=True), [1, 0, 2])
    for sp in classifier_data_old["spatial_patterns"]
]
fig, ax, cbar = plot_topo_class_pattern(
    spatial_patterns_rmse,
    np.arange(0.0, 0.81, 0.2),
    eeg_data_old["localizer"]["epochs"][1],
    row_break=1,
    figsize=(7, 1.5),
    cmap="Blues",
    vlim=[0.0, 15.0],
    data_scaler=1e6,
    return_handles=True,
    colorbar_label="",
    title_kwargs=dict(fontsize=0),
)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Localizer_SpatialPatterns_old.png"), dpi=300)
cbar.set_ticklabels("")
fig.savefig(Path(GRAPHICS_DIR, "Results", "Localizer_SpatialPatterns_old.svg"), dpi=300)


fig, ax = plt.subplots(figsize=(7, 3.5))
plot_sliding_classifier(
    classifier_data_young["performance"].mean(1),
    eeg_data_young["localizer"]["times"],
    chance=0.5,
    ax=ax,
)
ax.set_xlim([-0.2, 0.8])
ax.set_ylim([0.45, 0.7])
ax.set_yticks(np.arange(0.45, 0.71, 0.025))
ax.get_lines()[0].set_linewidth(2.0)
ax.get_lines()[1].set_linewidth(2.0)
ax.get_lines()[2].set_linewidth(2.5)
ax.legend().remove()
ax.tick_params(axis="both", length=8)
add_significance(
    ax=ax,
    stat_info=classifier_data_young["group_stats"]["cluster_dict"],
    times=eeg_data_young["localizer"]["times"],
    plot_type="linepoints",
    y_pos=0.475,
    color="k",
)
plt.tight_layout(pad=0.02)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Localizer_TimeCourse_young.png"), dpi=300)
ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
plt.tight_layout(pad=0.02)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Localizer_TimeCourse_young.svg"), dpi=300)


spatial_patterns_rmse = [
    np.transpose(np.std(sp, axis=0, keepdims=True), [1, 0, 2])
    for sp in classifier_data_young["spatial_patterns"]
]
fig, ax, cbar = plot_topo_class_pattern(
    spatial_patterns_rmse,
    np.arange(0.0, 0.81, 0.2),
    eeg_data_young["localizer"]["epochs"][1],
    row_break=1,
    figsize=(7, 1.5),
    cmap="Blues",
    vlim=[0.0, 15.0],
    data_scaler=1e6,
    return_handles=True,
    colorbar_label="",
    title_kwargs=dict(fontsize=0),
)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Localizer_SpatialPatterns_young.png"), dpi=300)
cbar.set_ticklabels("")
fig.savefig(Path(GRAPHICS_DIR, "Results", "Localizer_SpatialPatterns_young.svg"), dpi=300)


#######################################
"""
Resting and Cued Replay: Sequenceness

"""

fig, ax = plt.subplots(figsize=(7, 3))
plot_sequenceness(
    xcorr_data_old["resting"]["seq_net"],
    lags_ms,
    xcorr_data_old["resting"]["perm_net"],
    plot_dist_perm=None,
    plot_max_perm=False,
    plot_subs=False,
    ax=ax,
)
ax.set_xlim([0, MAX_LAG * 10])
ax.set_ylim([-0.04, 0.06])
ax.get_lines()[0].set_linewidth(2.0)
ax.get_lines()[1].set_linewidth(2.5)
thresh_t = xcorr_data_old["resting"]["threshold_data"]
ax.axhline(thresh_t, color="dimgrey", linestyle="--", linewidth=2.0)
ax.axhline(-thresh_t, color="dimgrey", linestyle="--", linewidth=2.0)
ax.set_xlabel("time lag (ms)", fontsize=12)
ax.set_ylabel("sequenceness (fwd − bwd)", fontsize=12)
ax.tick_params(axis="both", which="major", length=8)
ax.tick_params(axis="both", which="minor", length=4)
ax.legend().remove()
plt.tight_layout(pad=0.02)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Resting_Sequenceness_old.png"), dpi=300)
ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
plt.tight_layout(pad=0.02)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Resting_Sequenceness_old.svg"), dpi=300)


fig, ax = plt.subplots(figsize=(7, 3))
plot_sequenceness(
    xcorr_data_old["cued_replay"]["seq_net"],
    lags_ms,
    xcorr_data_old["cued_replay"]["perm_net"],
    plot_dist_perm=None,
    plot_max_perm=False,
    plot_subs=False,
    ax=ax,
)
ax.set_xlim([0, MAX_LAG * 10])
ax.set_ylim([-0.04, 0.06])
ax.get_lines()[0].set_linewidth(2.0)
ax.get_lines()[1].set_linewidth(2.5)
thresh_t = xcorr_data_old["cued_replay"]["threshold_data"]
ax.axhline(thresh_t, color="dimgrey", linestyle="--", linewidth=2.0)
ax.axhline(-thresh_t, color="dimgrey", linestyle="--", linewidth=2.0)
ax.set_xlabel("time lag (ms)", fontsize=12)
ax.set_ylabel("sequenceness (fwd − bwd)", fontsize=12)
ax.tick_params(axis="both", which="major", length=8)
ax.tick_params(axis="both", which="minor", length=4)
ax.legend().remove()
plt.tight_layout(pad=0.02)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "CuedReplay_Sequenceness_old.png"), dpi=300)
ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
plt.tight_layout(pad=0.02)
fig.savefig(Path(GRAPHICS_DIR, "Results", "CuedReplay_Sequenceness_old.svg"), dpi=300)


fig, ax = plt.subplots(figsize=(7, 3))
plot_sequenceness(
    xcorr_data_young["resting"]["seq_net"],
    lags_ms,
    xcorr_data_young["resting"]["perm_net"],
    plot_dist_perm=None,
    plot_max_perm=False,
    plot_subs=False,
    ax=ax,
)
ax.set_xlim([0, MAX_LAG * 10])
ax.set_ylim([-0.04, 0.06])
ax.get_lines()[0].set_linewidth(2.0)
ax.get_lines()[1].set_linewidth(2.5)
thresh_t = xcorr_data_young["resting"]["threshold_data"]
ax.axhline(thresh_t, color="dimgrey", linestyle="--", linewidth=2.0)
ax.axhline(-thresh_t, color="dimgrey", linestyle="--", linewidth=2.0)
ax.set_xlabel("time lag (ms)", fontsize=12)
ax.set_ylabel("sequenceness (fwd − bwd)", fontsize=12)
ax.tick_params(axis="both", which="major", length=8)
ax.tick_params(axis="both", which="minor", length=4)
ax.legend().remove()
plt.tight_layout(pad=0.02)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Resting_Sequenceness_young.png"), dpi=300)
ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
plt.tight_layout(pad=0.02)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Resting_Sequenceness_young.svg"), dpi=300)


fig, ax = plt.subplots(figsize=(7, 3))
plot_sequenceness(
    xcorr_data_young["cued_replay"]["seq_net"],
    lags_ms,
    xcorr_data_young["cued_replay"]["perm_net"],
    plot_dist_perm=None,
    plot_max_perm=False,
    plot_subs=False,
    ax=ax,
)
ax.set_xlim([0, MAX_LAG * 10])
ax.set_ylim([-0.04, 0.06])
ax.get_lines()[0].set_linewidth(2.0)
ax.get_lines()[1].set_linewidth(2.5)
thresh_t = xcorr_data_young["cued_replay"]["threshold_data"]
ax.axhline(thresh_t, color="dimgrey", linestyle="--", linewidth=2.0)
ax.axhline(-thresh_t, color="dimgrey", linestyle="--", linewidth=2.0)
ax.set_xlabel("time lag (ms)", fontsize=12)
ax.set_ylabel("sequenceness (fwd − bwd)", fontsize=12)
ax.tick_params(axis="both", which="major", length=8)
ax.tick_params(axis="both", which="minor", length=4)
ax.legend().remove()
plt.tight_layout(pad=0.02)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "CuedReplay_Sequenceness_young.png"), dpi=300)
ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
plt.tight_layout(pad=0.02)
fig.savefig(Path(GRAPHICS_DIR, "Results", "CuedReplay_Sequenceness_young.svg"), dpi=300)


fig, ax = plt.subplots(figsize=(7, 2.5))
ax.plot(
    lags_ms,
    xcorr_data_diff["resting"]["tmap"],
    color="steelblue",
    linewidth=2.5,
)
ax.axhline(
    xcorr_data_diff["resting"]["threshold"],
    color="dimgrey",
    linewidth=2.0,
    linestyle="--",
)
ax.axhline(0, color="k", linewidth=2.0)
ax.set_xlim([0, MAX_LAG * 10])
ax.set_ylim([-1.0, 4.0])
ax.set_ylabel("Two-sample t", fontsize=9)
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.tick_params(axis="both", which="major", length=8)
ax.tick_params(axis="both", which="minor", length=4)
ax.legend().remove()
ax.set_xlabel("Times (ms)", fontsize=9)
plt.tight_layout(pad=0.02)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Resting_Sequenceness_diff.png"), dpi=300)
ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
plt.tight_layout(pad=0.02)
fig.savefig(Path(GRAPHICS_DIR, "Results", "Resting_Sequenceness_diff.svg"), dpi=300)


fig, ax = plt.subplots(figsize=(7, 2.5))
ax.plot(
    lags_ms,
    xcorr_data_diff["cued_replay"]["tmap"],
    color="steelblue",
    linewidth=2.5,
)
ax.axhline(
    xcorr_data_diff["cued_replay"]["threshold"],
    color="dimgrey",
    linewidth=2.0,
    linestyle="--",
)
ax.axhline(0, color="k", linewidth=2.0)
ax.set_xlim([0, MAX_LAG * 10])
ax.set_ylim([-1.0, 4.0])
ax.set_ylabel("Two-sample t", fontsize=9)
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.tick_params(axis="both", which="major", length=8)
ax.tick_params(axis="both", which="minor", length=4)
ax.legend().remove()
ax.set_xlabel("Times (ms)", fontsize=9)
plt.tight_layout(pad=0.02)
plt.show(block=False)
fig.savefig(Path(GRAPHICS_DIR, "Results", "CuedReplay_Sequenceness_diff.png"), dpi=300)
ax.set(xlabel=None, ylabel=None, xticklabels=[], yticklabels=[])
plt.tight_layout(pad=0.02)
fig.savefig(Path(GRAPHICS_DIR, "Results", "CuedReplay_Sequenceness_diff.svg"), dpi=300)


#######################################
"""
Control Analyses

"""

effect_mask = np.where(
    xcorr_data_old["cued_replay"]["seq_net"].mean(0)
    > xcorr_data_old["cued_replay"]["threshold_data"]
)[0]

print(
    f"Time window of Sequenceness Effect: {lags_ms[effect_mask[0]]} - {lags_ms[effect_mask[-1]]} ms"
)

# Old: Sequenceness vs (Resting Length | Localizer Performance)
seq_net_effect_old = xcorr_data_old["cued_replay"]["seq_net"][:, effect_mask].mean(axis=-1)

r_stat, p_val = spearmanr(
    seq_net_effect_old,
    behavioral_data_old["resting"]["s_length"][:, 0],
)
print(f"Sequenceness vs Resting Length: rho = {r_stat:.3f}, p = {p_val:.3f}")

r_stat, p_val = spearmanr(seq_net_effect_old, classifier_data_old["performance"].mean(1)[:, 61])
print(f"Sequenceness vs Localizer Performance: rho = {r_stat:.3f}, p = {p_val:.3f}")


# Young: Sequenceness vs (Resting Length | Localizer Performance)
seq_net_effect_young = xcorr_data_young["cued_replay"]["seq_net"][:, effect_mask].mean(axis=-1)

r_stat, p_val = spearmanr(
    seq_net_effect_young,
    behavioral_data_young["resting"]["s_length"][:, 0],
)
print(f"Sequenceness vs Resting Length: rho = {r_stat:.3f}, p = {p_val:.3f}")

r_stat, p_val = spearmanr(seq_net_effect_young, classifier_data_young["performance"].mean(1)[:, 65])
print(f"Sequenceness vs Localizer Performance: rho = {r_stat:.3f}, p = {p_val:.3f}")


# Old: Sequenceness vs Individual Theta Peak Power
r_stat, p_val = spearmanr(seq_net_effect_old, freq_data_old["max_ch_peak_vals"])
print(f"Sequenceness vs Theta Peak Power: rho = {r_stat:.3f}, p = {p_val:.3f}")


# Sequenceness Effect Old: Before Learning vs Cue Period
wilcoxon_prerest = wilcoxon(
    xcorr_data_old["cued_replay"]["seq_net"][:, effect_mask].mean(1)
    - xcorr_data_old["preresting"]["seq_net"][:, effect_mask].mean(1),
    nan_policy="omit",
)
print(
    f"\nWilcoxon Test Before vs After Learning: W = {wilcoxon_prerest.statistic}, p = {wilcoxon_prerest.pvalue}\n"
)

# Sequenceness Effect Old: During Learning vs Cue Period
wilcoxon_seqlearn = wilcoxon(
    xcorr_data_old["cued_replay"]["seq_net"][:, effect_mask].mean(1)
    - xcorr_data_old["seq_learn"]["seq_net"][:, effect_mask].mean(1),
    nan_policy="omit",
)
print(
    f"\nWilcoxon Test During vs After Learning: W = {wilcoxon_seqlearn.statistic}, p = {wilcoxon_seqlearn.pvalue}\n"
)
