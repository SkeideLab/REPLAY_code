# -*- coding: utf-8 -*-
"""
Plot Results.

@author: Christopher Postzich
@github: mcpost
"""

# %% Imports

import os
import numpy as np
from scipy import stats
from skfda import FDataGrid
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.representation.basis import BSplineBasis
from utils.utils import hotelling_t2_test
from utils.plots import (
    plot_data_quant,
    plot_sliding_classifier,
    plot_topo_class_pattern,
    plot_sequenceness,
    plot_cond,
    add_signif_timepts,
    strip_axis_labels,
)
import matplotlib.pyplot as plt
%matplotlib qt


# %% Define necessary paths

code_dir = os.path.join('')
graph_dir = os.path.join('')


# %% Load Analysis Data
# Load data saved from load_transform_data.py

import pickle

results_dir = os.path.join(code_dir, '..', 'Results')
data_path = os.path.join(results_dir, 'analysis_data.pkl')

with open(data_path, 'rb') as f:
    data = pickle.load(f)

# Extract variables from loaded data
labels = data['labels']
transform_segments = data['transform_segments']
sample_mask = data['sample_mask']

# Participant info
participant_info = data['participant_info']

# EEG times and info
eeg_data = {'localizer': {'times': data['eeg_times']['localizer']}}
for seg in data['eeg_times'].keys():
    if seg not in eeg_data:
        eeg_data[seg] = {}
    eeg_data[seg]['times'] = data['eeg_times'][seg]

# Create a minimal epochs object for topographic plotting (if info was saved)
if data['eeg_epochs_info'] is not None:
    import mne
    eeg_data['localizer']['epochs'] = [mne.EpochsArray(
        np.zeros((1, len(data['eeg_epochs_info']['ch_names']), len(eeg_data['localizer']['times']))),
        data['eeg_epochs_info'],
        tmin=eeg_data['localizer']['times'][0]
    )]

# Classifier data
classifier_data = data['classifier_data']

# Decode data
decode_data = data['decode_data']

# TDLM data
tdlm_data = data['tdlm_data']
tdlm_data_non = data['tdlm_data_non']
tdlm_data_adv = data['tdlm_data_adv']
tdlm_data_nobl = data['tdlm_data_nobl']

# Cued replay data
cond_data = data['cond_data']
stim_pval = data['stim_pval']
perm_info = data['perm_info']

print(f"Data loaded from: {data_path}")


# %% Localizer: Time-resolved decoding

fig, ax = plt.subplots(figsize=(5,4))
plot_sliding_classifier(
    classifier_data['performance'].mean(1),
    eeg_data['localizer']['times'],
    perm_info=classifier_data['permutations'],
    sct_kwargs=dict(s=30, marker='o', color='k'),
    ax=ax
)
ax.set_xlim([-0.2,0.8])
ax.get_lines()[0].set_linewidth(1.5)
# strip_axis_labels(ax)
ax.tick_params(axis='both', length=5)
plt.tight_layout(pad=0.02)
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_Localizer_Decoding_TimeCourse.svg'
# ), dpi=300)


# %% Localizer: Spatial Patterns for decoding

spatial_patterns_rmse = [
    np.std(sp, axis=1, keepdims=True)
    for sp in classifier_data['spatial_patterns']
]
fig, ax, cbar = plot_topo_class_pattern(
    spatial_patterns_rmse,
    np.arange(0.0,0.7,0.05),
    eeg_data['localizer']['epochs'][0],
    row_break=2,
    figsize=(12,5),
    cmap='Blues',
    vlim=[0.0, 15.0],
    data_scaler=1e6,
    return_handles=True,
    colorbar_label="",
    title_kwargs=dict(color='w', fontsize=18)
)
#cbar.set_ticklabels('')
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_Localizer_Decoding_SpatPatterns_RMSE_Timeseries.svg'
# ), dpi=300)


# %% Localizer: Time-resolved decoding Single Participant

fig, axes = plt.subplots(
    nrows=4,
    ncols=5,
    subplot_kw={},
    gridspec_kw={
        'height_ratios': [1]*4,
        'width_ratios': [1]*5,
        'hspace': 0.3,
        'wspace': 0.1
    },
    figsize=(15,9)
)
for i, ax in enumerate(axes.flat):
    ax.axhline(0.5, color="k", linestyle="--", label="Chance Level")
    ax.axvline(0.0, color="k", linestyle="-")
    ax.plot(
        eeg_data['localizer']['times'],
        classifier_data['performance'][i,:,:].mean(0),
        label="Class. Perf."
    )
    ax.fill_between(
        eeg_data['localizer']['times'],
        classifier_data['performance'][i,:,:].mean(0) - classifier_data['performance'][i,:,:].std(0),
        classifier_data['performance'][i,:,:].mean(0) + classifier_data['performance'][i,:,:].std(0),
        alpha=0.2
    )
    ax.set_title(f'Participant {i+1} - {participant_info["info_names"][i]}')
    ax.set_ylim([0.3, 0.8])
    ax.set_xlim([-0.2, 0.8])
    ax.set_xticks(np.arange(-0.2,0.81,0.2))
    # strip_axis_labels(ax)
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_Localizer_Decoding_TimeCourse_SingleParticipant.svg'
# ), dpi=300)


# %% Resting: Sequenceness Measure

fig, ax = plt.subplots(figsize=(7,4))
ax.axvline(30, color="k", linestyle="dotted")
ax.axvline(60, color="k", linestyle="dotted")
plot_sequenceness(
    np.diff(tdlm_data['resting']['sequenceness'][:,1::-1,:], 1, 1).squeeze(),
    np.arange(10,(tdlm_data['max_lag']+1)*10,10),
    np.diff(tdlm_data['resting']['permutations'][:,:,1::-1,:], 1, 2).squeeze(),
    plot_dist_perm=None,
    plot_max_perm='max',
    plot_subs=False,
    ax=ax
)
ax.set_xlim([0,300])
ax.set_ylim([-0.4,0.4])
ax.get_lines()[-1].set_linewidth(1.5)
ax.get_lines()[2].set_linewidth(1.0)
ax.get_lines()[3].set_linewidth(1.0)
# strip_axis_labels(ax)
ax.tick_params(axis='both', which='major', length=5)
plt.tight_layout(pad=0.05)
ax.legend().remove()
axins = fig.add_axes([0.37, 0.62, 0.25, 0.32])
plot_sequenceness(
    np.diff(tdlm_data['resting']['sequenceness'][:,1::-1,:], 1, 1).squeeze(),
    np.arange(10,(tdlm_data['max_lag']+1)*10,10),
    np.diff(tdlm_data['resting']['permutations'][:,:,1::-1,:], 1, 2).squeeze(),
    plot_dist_perm=97.5,
    plot_max_perm='max',
    plot_subs=False,
    ax=axins
)
axins.set_xlim([20,80])
axins.set_ylim([0.28,0.37])
# strip_axis_labels(axins)
axins.legend().remove()
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_Resting_Sequenceness_GA.svg'
# ), dpi=300)


# %% Resting: Statistical Test

resting_effect = np.mean(np.diff(tdlm_data['resting']['sequenceness'][:,1::-1,4:6], 1, 1).squeeze(),0)

perm_dist = np.mean(np.diff(tdlm_data['resting']['permutations'][:,:,1::-1,:], 1, 2).squeeze(), 0)
max_perms = np.max(np.abs(perm_dist),1).flatten()

zvals = (resting_effect - max_perms.mean()) / max_perms.std()

print(f"\nPreresting Break Effect at 50 ms: z = {zvals[0]}, p = {1-stats.norm.cdf(zvals[0])}\n")
print(f"\nPreresting Break Effect at 60 ms: z = {zvals[1]}, p = {1-stats.norm.cdf(zvals[1])}\n")


# %% Resting: Sequenceness Measure Empirical Transition Matrices

emp_tm = np.reshape(
    tdlm_data['resting']['empirical_tm'],
    (tdlm_data['max_lag'], len(tdlm_data['classes']), len(tdlm_data['classes'])),
    order='F'
)

# Visualizations
fig, axes = plt.subplots(
    5,
    10,
    figsize=(12, 8)
)

for i,ax in enumerate(axes.flat):
    ax.axis('off')
    if emp_tm.shape[0] > i:
        ax.imshow(
            emp_tm[i,:,:],
            vmin=np.percentile(emp_tm.flatten(), 2.5),
            vmax=np.percentile(emp_tm.flatten(), 97.5),
            cmap='Blues',
            aspect='equal'
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{(i+1)*10} ms', {'fontsize': 10})
plt.tight_layout(pad=0.05, h_pad=0.2, w_pad=0.6)
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_Resting_Sequenceness_EmpiricalTMs.svg'
# ))


# %% Resting: Sequenceness Measure Detailed

fig, ax = plt.subplots(
    nrows=4,
    ncols=1,
    figsize=(6,8)
)

for i,label in enumerate(['Forward','Backward','Auto','Intercept']):
    plot_sequenceness(
            tdlm_data['resting']['sequenceness'][:,i,:],
            np.arange(10,(tdlm_data_non['max_lag']+1)*10,10),
            tdlm_data['resting']['permutations'][:,:,i,:],
            plot_dist_perm=None,
            plot_max_perm='max',
            plot_subs=False,
            ax=ax[i]
        )
    # strip_axis_labels(ax[i])
    ax[i].set_xlim([0, 300])
    ax[i].set_title(label)
    ax[i].legend().remove()
    ax[i].get_lines()[-1].set_linewidth(1.5)
    ax[i].tick_params(axis='both', which='major', length=5)
plt.tight_layout(pad=0.05, h_pad=2)
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_Resting_Sequenceness_Detailed.svg'
# ))


# %% Resting: Sequenceness Measure Detailed (No Binning)

fig, ax = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(6,7)
)
for i,label in enumerate(['resting','preresting','preresting_break']):
    skip_nan_idx = ~np.isnan(tdlm_data[label]['sequenceness'][:,1,1])
    plot_sequenceness(
            tdlm_data_non[label]['sequenceness'][skip_nan_idx,2,:],
            np.arange(10,(tdlm_data_non['max_lag']+1)*10,10),
            tdlm_data_non[label]['permutations'][skip_nan_idx,:,2,:],
            plot_dist_perm=None,
            plot_max_perm=None,
            plot_subs=False,
            ax=ax[i]
        )
    ax[i].set_ylim([-0.1,1.0])
    # strip_axis_labels(ax[i])
    ax[i].set_xlim([0, 300])
    ax[i].set_title(label)
    ax[i].legend().remove()
    ax[i].get_lines()[-1].set_linewidth(1.5)
    ax[i].tick_params(axis='both', which='major', length=5)
plt.tight_layout(pad=0.05, h_pad=4)
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_Resting_Sequenceness_NoOsc_Detailed.svg'
# ))


# %% Resting: Sequenceness Measure by Participant (Resting Time)

fig, ax = plt.subplots(figsize=(5,2))
ax.axvline(30, color="k", linestyle="dotted")
ax.axvline(60, color="k", linestyle="dotted")
_,cbar = plot_sequenceness(
    -np.diff(tdlm_data['resting']['sequenceness'][:,0:2,:], 1, 1).squeeze(),
    np.arange(10,(tdlm_data['max_lag']+1)*10,10),
    -np.diff(tdlm_data['resting']['permutations'][:,:,0:2,:], 1, 2).squeeze(),
    plot_dist_perm=None,
    plot_max_perm=None,
    ax=ax,
    return_cbar_handle=True,
    plot_subs=dict(
        color_values = participant_info['time_rest'],
        cmap = 'cool',
        cbar_title = ''
    )
)
ax.set_xlim([0, 300])
ax.set_ylim((-0.7, 0.7))
# strip_axis_labels(ax)
ax.tick_params(axis='both', which='major', length=5)
plt.tight_layout(pad=0.05)
# ax.legend().remove()
cbar.set_ticks(np.arange(20,200,20))
# cbar.set_ticklabels('')
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_Resting_Sequenceness_Sub_RestingTime.svg'
# ))


# %% Resting: Sequenceness Measure by Participant (Age)

fig, ax = plt.subplots(figsize=(5,2))
ax.axvline(30, color="k", linestyle="dotted")
ax.axvline(60, color="k", linestyle="dotted")
_,cbar = plot_sequenceness(
    -np.diff(tdlm_data['resting']['sequenceness'][:,0:2,:], 1, 1).squeeze(),
    np.arange(10,(tdlm_data['max_lag']+1)*10,10),
    -np.diff(tdlm_data['resting']['permutations'][:,:,0:2,:], 1, 2).squeeze(),
    plot_dist_perm=None,
    plot_max_perm=None,
    ax=ax,
    return_cbar_handle=True,
    plot_subs=dict(
        color_values = participant_info['participant_included']['Age_Months'].to_numpy(),
        cmap = 'cool',
        cbar_title = ''
    )
)
ax.set_xlim([0, 300])
ax.set_ylim((-0.7, 0.7))
# strip_axis_labels(ax)
ax.tick_params(axis='both', which='major', length=5)
plt.tight_layout(pad=0.05)
# ax.legend().remove()
cbar.set_ticks(np.arange(10,14,1))
# cbar.set_ticklabels('')
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_Resting_Sequenceness_Sub_Age.svg'
# ))


# %% Before Learning: Before Localizer Break

skip_nan_idx = ~np.isnan(tdlm_data['preresting']['sequenceness'][:,1,1])
fig, ax = plt.subplots(figsize=(7,4))
ax.axvline(30, color="k", linestyle="dotted")
ax.axvline(60, color="k", linestyle="dotted")
plot_sequenceness(
    np.diff(tdlm_data['preresting']['sequenceness'][skip_nan_idx,1::-1,:], 1, 1).squeeze(),
    np.arange(10,(tdlm_data['max_lag']+1)*10,10),
    np.diff(tdlm_data['preresting']['permutations'][skip_nan_idx,:,1::-1,:], 1, 2).squeeze(),
    plot_dist_perm=97.5,
    plot_max_perm='max',
    plot_subs=False,
    ax=ax
)
ax.set_ylim([-0.4, 0.4])
ax.set_xlim([0, 300])
ax.get_lines()[-1].set_linewidth(1.5)
ax.get_lines()[2].set_linewidth(1.0)
ax.get_lines()[3].set_linewidth(1.0)
# strip_axis_labels(ax)
# ax.legend().remove()
ax.tick_params(axis='both', which='major', length=5)
axins = fig.add_axes([0.37, 0.66, 0.25, 0.28])
plot_sequenceness(
    np.diff(tdlm_data['preresting']['sequenceness'][skip_nan_idx,1::-1,:], 1, 1).squeeze(),
    np.arange(10,(tdlm_data['max_lag']+1)*10,10),
    np.diff(tdlm_data['preresting']['permutations'][skip_nan_idx,:,1::-1,:], 1, 2).squeeze(),
    plot_dist_perm=97.5,
    plot_max_perm='max',
    plot_subs=False,
    ax=axins
)
#axins.get_lines()[0].set_linewidth(1.0)
axins.set_xlim([20,80])
axins.set_ylim([0.145,0.21])
# strip_axis_labels(axins)
axins.legend().remove()
plt.tight_layout(pad=0.05)
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_PreResting_Sequenceness_GA.svg'
# ), dpi=300)


# %% PreResting: Statistical Tests

# Comparing Before Learning to after Learning
prerest_effect = np.mean(np.diff(tdlm_data['preresting']['sequenceness'][skip_nan_idx,1::-1,4:6], 1, 1).squeeze(),0)

perm_dist = np.mean(np.diff(tdlm_data['preresting']['permutations'][skip_nan_idx,:,1::-1,:], 1, 2).squeeze(), 0)
max_perms = np.max(np.abs(perm_dist),1).flatten()

zvals = (prerest_effect - max_perms.mean()) / max_perms.std()
print(f"\nPreresting Effect at 50 ms: z = {zvals[0]}, p = {1-stats.norm.cdf(zvals[0])}\n")
print(f"\nPreresting Effect at 60 ms: z = {zvals[1]}, p = {1-stats.norm.cdf(zvals[1])}\n")


# Before Learning Before Localizer Break
rest_peak = np.diff(tdlm_data['resting']['sequenceness'][:,1::-1,4:6], 1, 1).mean(-1).squeeze()
prerest_peak = np.diff(tdlm_data['preresting']['sequenceness'][:,1::-1,4:6], 1, 1).mean(-1).squeeze()

wilcoxon_prerest = stats.wilcoxon(
    rest_peak - prerest_peak,
    nan_policy='omit',
)
print(f"\nWilcoxon Test Prebreak: W = {wilcoxon_prerest.statistic}, p = {wilcoxon_prerest.pvalue}\n")


# %% Functional Data Analysis: Preresting vs Resting

# Comparing Before Learning to after Learning - Functional ANOVA
# Create two time series with different shapes
time_idx = np.arange(1,8,1)
time = (time_idx+1)*10
timeseries1 = np.diff(tdlm_data['resting']['sequenceness'][skip_nan_idx,:,:][:,1::-1,time_idx], 1, 1).squeeze()
timeseries2 = np.diff(tdlm_data['preresting']['sequenceness'][skip_nan_idx,:,:][:,1::-1,time_idx], 1, 1).squeeze()

# Convert to functional data objects
fd1 = FDataGrid(timeseries1, grid_points=time)
fd2 = FDataGrid(timeseries2, grid_points=time)

# Optional: Smooth the data with B-splines for better derivative estimation
basis = BSplineBasis(n_basis=8, domain_range=(0, 1))
smoother = BasisSmoother(basis)
fd1_smooth = smoother.fit_transform(fd1)
fd2_smooth = smoother.fit_transform(fd2)

# Perform tests
print("=" * 60)
print("FUNCTIONAL DATA ANALYSIS TESTS")
print("=" * 60)

# Test Original curves (levels)
stat_original, f_original, p_original, df = hotelling_t2_test(fd1_smooth, fd2_smooth)
print(f"\n1. TEST OF CURVE LEVELS (Original Functions)")
print(f"   H₀: The two mean curves are identical")
print(f"   Hotelling's T² statistic: {stat_original:.3f}")
print(f"   F({df[0]}, {df[1]}): {f_original:.3f}")
print(f"   P-value: {p_original:.4f}")
print(f"   Result: {'SIGNIFICANT' if p_original < 0.05 else 'NOT SIGNIFICANT'} at α=0.05")

print("\n" + "=" * 60)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# Row 1: Original curves
fd1_smooth.plot(axes=axes[0], color='blue', alpha=0.3, linewidth=0.8)
fd2_smooth.plot(axes=axes[0], color='red', alpha=0.3, linewidth=0.8)
fd1_smooth.mean().plot(axes=axes[0], color='blue', linewidth=3)
fd2_smooth.mean().plot(axes=axes[0], color='red', linewidth=3)
axes[0].set_title('Mean + Individual Curves')
axes[0].set_xlabel('lags (ms)')
axes[0].set_ylabel('sequenceness')
axes[0].legend()
axes[0].grid(True, alpha=0.3)


# Row 2: Mean difference
diff_mean = fd1_smooth.mean() - fd2_smooth.mean()
diff_mean.plot(axes=axes[1], color='purple', linewidth=2.5)
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[1].fill_between(time, 0, diff_mean.data_matrix.squeeze(),
                alpha=0.3, color='purple')
axes[1].set_title('Difference Between Mean Curves')
axes[1].set_xlabel('lags (ms)')
axes[1].set_ylabel('Difference')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %% Before Learning: After Localizer Break

fig, ax = plt.subplots(figsize=(7,4))
ax.axvline(30, color="k", linestyle="dotted")
ax.axvline(60, color="k", linestyle="dotted")
plot_sequenceness(
    np.diff(tdlm_data['preresting_break']['sequenceness'][:,1::-1,:], 1, 1).squeeze(),
    np.arange(10,(tdlm_data['max_lag']+1)*10,10),
    np.diff(tdlm_data['preresting_break']['permutations'][:,:,1::-1,:], 1, 2).squeeze(),
    plot_dist_perm=97.5,
    plot_max_perm='max',
    plot_subs=False,
    ax=ax
)
ax.set_ylim([-0.4, 0.4])
ax.set_xlim([0, 300])
ax.get_lines()[-1].set_linewidth(1.5)
ax.get_lines()[2].set_linewidth(1.0)
ax.get_lines()[3].set_linewidth(1.0)
# strip_axis_labels(ax)
# ax.legend().remove()
ax.tick_params(axis='both', which='major', length=5)
axins = fig.add_axes([0.37, 0.68, 0.25, 0.28])
plot_sequenceness(
    np.diff(tdlm_data['preresting_break']['sequenceness'][:,1::-1,:], 1, 1).squeeze(),
    np.arange(10,(tdlm_data['max_lag']+1)*10,10),
    np.diff(tdlm_data['preresting_break']['permutations'][:,:,1::-1,:], 1, 2).squeeze(),
    plot_dist_perm=97.5,
    plot_max_perm='max',
    plot_subs=False,
    ax=axins
)
axins.set_xlim([20,80])
axins.set_ylim([0.10,0.21])
# strip_axis_labels(axins)
axins.legend().remove()
plt.tight_layout(pad=0.05)
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_PreRestingBreak_Sequenceness_GA.svg'
# ), dpi=300)


# %% PreResting Break: Statistical Tests

# Comparing Before Learning to after Learning
prerestbreak_effect = np.mean(np.diff(tdlm_data['preresting_break']['sequenceness'][:,1::-1,4:6], 1, 1).squeeze(),0)

perm_dist = np.mean(np.diff(tdlm_data['preresting_break']['permutations'][:,:,1::-1,:], 1, 2).squeeze(), 0)
max_perms = np.max(np.abs(perm_dist),1).flatten()

zvals = (prerestbreak_effect - max_perms.mean()) / max_perms.std()

print(f"\nPreresting Break Effect at 50 ms: z = {zvals[0]}, p = {1-stats.norm.cdf(zvals[0])}\n")
print(f"\nPreresting Break Effect at 60 ms: z = {zvals[1]}, p = {1-stats.norm.cdf(zvals[1])}\n")


# Before Learning After Localizer Break
rest_peak = np.diff(tdlm_data['resting']['sequenceness'][:,1::-1,4:6], 1, 1).mean(-1).squeeze()
prerestbreak_peak = np.diff(tdlm_data['preresting_break']['sequenceness'][:,1::-1,4:6], 1, 1).mean(-1).squeeze()

wilcoxon_prerest_break = stats.wilcoxon(
    rest_peak - prerestbreak_peak,
)
print(f"\nWilcoxon Test Prerest break: W = {wilcoxon_prerest_break.statistic}, p = {wilcoxon_prerest_break.pvalue}\n")


# %% Functional Data Analysis: PreResting Break vs Resting

# Comparing Before Learning to after Learning - Functional ANOVA
# Create two time series with different shapes
time_idx = np.arange(1,8,1)
time = (time_idx+1)*10
timeseries1 = np.diff(tdlm_data['resting']['sequenceness'][:,:,:][:,1::-1,time_idx], 1, 1).squeeze()
timeseries2 = np.diff(tdlm_data['preresting_break']['sequenceness'][:,:,:][:,1::-1,time_idx], 1, 1).squeeze()

# Convert to functional data objects
fd1 = FDataGrid(timeseries1, grid_points=time)
fd2 = FDataGrid(timeseries2, grid_points=time)

# Optional: Smooth the data with B-splines for better derivative estimation
basis = BSplineBasis(n_basis=8, domain_range=(0, 1))
smoother = BasisSmoother(basis)
fd1_smooth = smoother.fit_transform(fd1)
fd2_smooth = smoother.fit_transform(fd2)

# Perform tests
print("=" * 60)
print("FUNCTIONAL DATA ANALYSIS TESTS")
print("=" * 60)

# Test Original curves (levels)
stat_original, f_original, p_original, df = hotelling_t2_test(fd1_smooth, fd2_smooth)
print(f"\n1. TEST OF CURVE LEVELS (Original Functions)")
print(f"   H₀: The two mean curves are identical")
print(f"   Hotelling's T² statistic: {stat_original:.3f}")
print(f"   F({df[0]}, {df[1]}): {f_original:.3f}")
print(f"   P-value: {p_original:.4f}")
print(f"   Result: {'SIGNIFICANT' if p_original < 0.05 else 'NOT SIGNIFICANT'} at α=0.05")

print("\n" + "=" * 60)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# Row 1: Original curves
fd1_smooth.plot(axes=axes[0], color='blue', alpha=0.3, linewidth=0.8)
fd2_smooth.plot(axes=axes[0], color='red', alpha=0.3, linewidth=0.8)
fd1_smooth.mean().plot(axes=axes[0], color='blue', linewidth=3)
fd2_smooth.mean().plot(axes=axes[0], color='red', linewidth=3)
axes[0].set_title('Mean + Individual Curves')
axes[0].set_xlabel('lags (ms)')
axes[0].set_ylabel('sequenceness')
axes[0].legend()
axes[0].grid(True, alpha=0.3)


# Row 2: Mean difference
diff_mean = fd1_smooth.mean() - fd2_smooth.mean()
diff_mean.plot(axes=axes[1], color='purple', linewidth=2.5)
axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
axes[1].fill_between(time, 0, diff_mean.data_matrix.squeeze(),
                alpha=0.3, color='purple')
axes[1].set_title('Difference Between Mean Curves')
axes[1].set_xlabel('lags (ms)')
axes[1].set_ylabel('Difference')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %% Sequence Learning: Sequenceness Measure

fig, ax = plt.subplots(figsize=(7,4))
ax.axvline(30, color="k", linestyle="dotted")
ax.axvline(60, color="k", linestyle="dotted")
plot_sequenceness(
    np.diff(tdlm_data['seq_learn']['sequenceness'][:,1::-1,:], 1, 1).squeeze(),
    np.arange(10,(tdlm_data['max_lag']+1)*10,10),
    np.diff(tdlm_data['seq_learn']['permutations'][:,:,1::-1,:], 1, 2).squeeze(),
    plot_dist_perm=97.5,
    plot_max_perm='max',
    plot_subs=False,
    ax=ax
)
ax.set_ylim([-0.5, 0.5])
ax.set_xlim([0, 300])
ax.get_lines()[-1].set_linewidth(1.5)
strip_axis_labels(ax)
ax.legend().remove()
ax.tick_params(axis='both', which='major', length=5)
plt.tight_layout(pad=0.05)
axins = fig.add_axes([0.36, 0.67, 0.20, 0.27])
plot_sequenceness(
    np.diff(tdlm_data['seq_learn']['sequenceness'][:,1::-1,:], 1, 1).squeeze(),
    np.arange(10,(tdlm_data['max_lag']+1)*10,10),
    np.diff(tdlm_data['seq_learn']['permutations'][:,:,1::-1,:], 1, 2).squeeze(),
    plot_dist_perm=97.5,
    plot_max_perm='max',
    plot_subs=False,
    ax=axins
)
axins.set_xlim([20,80])
axins.set_ylim([0.25,0.4])
strip_axis_labels(axins)
axins.legend().remove()
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_SequenceLearn_Sequenceness_GA.svg'
# ), dpi=300)


# %% No Baseline: Sequenceness Measure

fig, axes = plt.subplots(3, 1, figsize=(7,8))
for ax,name in zip(axes.flat, ['resting', 'preresting', 'preresting_break']):
    skip_nan_idx = ~np.isnan(tdlm_data_nobl[name]['sequenceness'][:,1,1])
    ax.axvline(30, color="k", linestyle="dotted")
    ax.axvline(60, color="k", linestyle="dotted")
    plot_sequenceness(
        np.diff(tdlm_data_nobl[name]['sequenceness'][skip_nan_idx,1::-1,:], 1, 1).squeeze(),
        np.arange(10,(tdlm_data_nobl['max_lag']+1)*10,10),
        np.diff(tdlm_data_nobl[name]['permutations'][skip_nan_idx,:,1::-1,:], 1, 2).squeeze(),
        plot_dist_perm=97.5,
        plot_max_perm='max',
        plot_subs=False,
        ax=ax
    )
    ax.set_ylim([-0.2,0.2])
    ax.set_xlim([0, 300])
    ax.get_lines()[-1].set_linewidth(1.5)
    # strip_axis_labels(ax)
    ax.tick_params(axis='both', which='major', length=5)
    ax.legend().remove()
plt.tight_layout(pad=0.05, h_pad=3.5)
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_Resting_Sequenceness_NoBaseline.svg'
# ), dpi=300)


# %% No Baseline: Permutation Matrices

fig, axes = plt.subplots(1, 4, figsize=(8,2))
tdlm_data_nobl['tdlm'].plot_permutation_matrices(
    axes=axes
)
plt.tight_layout(pad=0.2, w_pad=2, h_pad=2)
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_Resting_Sequenceness_NoBaseline_PermMatrices.svg'
# ), dpi=300)


# %% Full Transition Matrix: Sequenceness Measure

fig, axes = plt.subplots(3, 1, figsize=(7,8))
for ax,name in zip(axes.flat, ['resting', 'preresting', 'preresting_break']):
    skip_nan_idx = ~np.isnan(tdlm_data_adv[name]['sequenceness'][:,1,1])
    ax.axvline(30, color="k", linestyle="dotted")
    ax.axvline(60, color="k", linestyle="dotted")
    plot_sequenceness(
        np.diff(tdlm_data_adv[name]['sequenceness'][skip_nan_idx,1::-1,:], 1, 1).squeeze(),
        np.arange(10,(tdlm_data_adv['max_lag']+1)*10,10),
        np.diff(tdlm_data_adv[name]['permutations'][skip_nan_idx,:,1::-1,:], 1, 2).squeeze(),
        plot_dist_perm=97.5,
        plot_max_perm='max',
        plot_subs=False,
        ax=ax
    )
    ax.set_ylim([-0.4,0.4])
    ax.set_xlim([0, 300])
    ax.get_lines()[-1].set_linewidth(1.5)
    # strip_axis_labels(ax)
    ax.tick_params(axis='both', which='major', length=5)
    ax.legend().remove()
plt.tight_layout(pad=0.05, h_pad=3.5)
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_Resting_Sequenceness_FullTransition.svg'
# ), dpi=300)


# %% Full Transition Matrix: Permutation Matrices

fig, axes = plt.subplots(1, 4, figsize=(8,2))
tdlm_data_adv['tdlm'].plot_permutation_matrices(
    axes=axes
)
plt.tight_layout(pad=0.2, w_pad=2, h_pad=2)
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_Resting_Sequenceness_FullTransition_PermMatrices.svg'
# ), dpi=300)


# %% Resting: Sequenceness Replay Example 1

tdlm = tdlm_data['tdlm']

X = decode_data['resting']['probabilities'][5].transpose(1, 0, 2).reshape(4, -1)
replay_onsets = tdlm.replay_onsets(X, lag=5, model_tm=tdlm_data['tm'])

vmin = stats.norm.ppf(0.87)
vmax = stats.norm.ppf(0.99)

ex = 0
startind = replay_onsets[ex]-10
stopind = replay_onsets[ex]+21

fig, ax = plt.subplots(figsize=(5, 2.5))
dat = X[:, startind:stopind]
dat_z = (dat - dat.mean(1, keepdims=True)) / dat.std(1, keepdims=True)
ax.imshow(dat_z, vmin=vmin, vmax=vmax, cmap='hot', aspect='auto')
# strip_axis_labels(ax)
ax.tick_params(axis='both',length=0)
plt.tight_layout(pad=0.05)
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_Resting_Sequenceness_ReplayExample1.svg'
# ))


# %% Resting: Sequenceness Replay Example 2

X = decode_data['resting']['probabilities'][12].transpose(1, 0, 2).reshape(4, -1)
replay_onsets = tdlm.replay_onsets(X, lag=5, model_tm=tdlm_data['tm'])

vmin = stats.norm.ppf(0.85)
vmax = stats.norm.ppf(0.99)

ex = 42
startind = replay_onsets[ex]-10
stopind = replay_onsets[ex]+21

fig, ax = plt.subplots(figsize=(5, 2.5))
dat = X[:, startind:stopind]
dat_z = (dat - dat.mean(1, keepdims=True)) / dat.std(1, keepdims=True)
ax.imshow(dat_z, vmin=vmin, vmax=vmax, cmap='hot', aspect='auto')
# strip_axis_labels(ax)
ax.tick_params(axis='both',length=0)
plt.tight_layout(pad=0.05)
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_Resting_Sequenceness_ReplayExample2.svg'
# ))


# %% Resting: Sequenceness Replay Example 3

X = decode_data['resting']['probabilities'][18].transpose(1, 0, 2).reshape(4, -1)
replay_onsets = tdlm.replay_onsets(X, lag=5, model_tm=tdlm_data['tm'])

vmin = stats.norm.ppf(0.87)
vmax = stats.norm.ppf(0.99)

ex = 13
startind = replay_onsets[ex]-10
stopind = replay_onsets[ex]+21

fig, ax = plt.subplots(figsize=(5, 2.5))
dat = X[:, startind:stopind]
dat_z = (dat - dat.mean(1, keepdims=True)) / dat.std(1, keepdims=True)
ax.imshow(dat_z, vmin=vmin, vmax=vmax, cmap='hot', aspect='auto')
# strip_axis_labels(ax)
ax.tick_params(axis='both',length=0)
plt.tight_layout(pad=0.05)
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_Resting_Sequenceness_ReplayExample3.svg'
# ))


# %% Resting: Sequenceness Replay Onsets over time

# Normalize all event times to [0, 1] interval
normalized_times = []
for onsets, length in zip(
        tdlm_data['resting']['replay_onsets'],
        tdlm_data['resting']['num_timepoints']
    ):
    if len(onsets) > 0 and length > 0:
        normalized = onsets / length
        normalized_times.extend(normalized)

normalized_times = np.array(normalized_times)

# Kolmogorov-Smirnov test against uniform distribution
ks_stat, ks_pvalue = stats.kstest(normalized_times, 'uniform')

print(f"\nKolmogrov Smirnoff Test: D = {ks_stat}, p = {ks_pvalue}\n")

# Visualizations
fig, ax = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

# Histogram
density, bins = np.histogram(normalized_times, 20, density=True)
unity_density = density / density.sum()
widths = bins[:-1] - bins[1:]
ax[0].bar(bins[1:], unity_density, width=widths, alpha=0.7, edgecolor='black')
ax[0].axhline(1/len(bins), color='red', linestyle='--', label='Uniform expectation')
ax[0].set_xlabel('') # Normalized Time
ax[0].set_ylabel('') # Density
# strip_axis_labels(ax[0], xlabel=False, ylabel=False)

# Q-Q plot
sorted_times = np.sort(normalized_times)
theoretical_quantiles = np.linspace(0, 1, len(sorted_times))
ax[1].scatter(theoretical_quantiles, sorted_times, alpha=0.5, s=10)
ax[1].plot([0, 1], [0, 1], 'r--')
ax[1].set_ylim([0, 1])
ax[1].set_xlim([0, 1])
ax[1].set_xlabel('') # Theoretical Quantiles (Uniform)
ax[1].set_ylabel('') # Observed Quantiles
# strip_axis_labels(ax[1], xlabel=False, ylabel=False)

plt.tight_layout(pad=0.05, w_pad=7)
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_Resting_Sequenceness_ReplayOnsetStatistics.svg'
# ))


# %% Cued Replay: Reactivation

fig, ax = plt.subplots()
plot_cond(cond_data,
          eeg_data['cued_replay']['times'],
          ['apple','chair','face'],
          axes=ax,
          axhline_kwargs = dict(y=0, color='k', linestyle='--'))
ax.set_xlim([-0.2, 1.0])
ax.set_ylim([-0.054, 0.154])
for i,labl in enumerate(['apple','chair','face']):
    perm_info['stat_data'] = stim_pval[i][:,0]
    perm_info['corrected_pvals'] = stim_pval[i][:,1]
    perm_info['mask'] = stim_pval[i][:,2]
    add_signif_timepts(ax, perm_info, plot_type='linepoints',
                      label=labl, use_pval_alpha=False,
                      y_pos=-0.01-0.007*i)
ax.set_xlabel('')
ax.set_ylabel('')
# strip_axis_labels(ax, xticklabels=False, yticklabels=False)
plt.tight_layout(pad=0.05)
# fig.savefig(os.path.join(
#     graph_dir,
#     'Things_CuedReplay_Timecours_clusterdepth.svg'
# ))


# %% Participant Data Overview

plot_data_quant(
    participant_info['loc_trialnum'],
    participant_info['time_rest'],
    labels,
    participant_info['info_names'],
    colors = ('#1f77b4ff','#ff7f0eff','#2ca02cff'),
    # save = os.path.join(
    #     graph_dir,
    #     f'Things_DataQuanitification.svg'
    # )
   )
