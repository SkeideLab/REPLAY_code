from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from utils.decode import run_decode
from utils.freq import run_freq_analysis
from utils.imports import Things_Importer, load_participant_overview
from utils.paths import RAW_DIR, INFO_DIR, PREPROC_DIR
from utils.xcorr import run_contrasts, run_group_comparison, run_xcorr
from utils.xdecode import run_xdecode

# Which data segments to analyse
SEGMENTS = [
    "resting",
    "cued_replay",
    "seq_learn",
    "preresting",
]

# Import Participant Information
participant_overview = load_participant_overview(INFO_DIR)

# Which data segments to import
load_segments = {
    "localizer": {
        "load": True,
        "path": Path("Segments", "Localizer"),
        "filename_suffix": "_Epochs.fif",
        "process_labels": True,
        "reshape_data": False,
        "resample_freq": None,
        "filter_params": {
            "l_freq": 1,
            "h_freq": 30,
            "filter_kwargs": {"method": "fir", "verbose": False},
            "decimate": 10,
        },
    },
    "resting": {
        "load": True,
        "path": Path("Segments", "Resting"),
        "filename_suffix": "_Epochs.fif",
        "process_labels": False,
        "reshape_data": False,
        "resample_freq": None,
        "filter_params": {
            "l_freq": 1,
            "h_freq": 30,
            "filter_kwargs": {"method": "fir", "verbose": False},
            "decimate": 10,
        },
    },
    "cued_replay": {
        "load": True,
        "path": Path("Segments", "CuedReplay"),
        "filename_suffix": "_Epochs.fif",
        "process_labels": False,
        "reshape_data": False,
        "resample_freq": None,
        "filter_params": {
            "l_freq": 1,
            "h_freq": 30,
            "filter_kwargs": {"method": "fir", "verbose": False},
            "decimate": 10,
        },
    },
    "seq_learn": {
        "load": True,
        "path": Path("Segments", "LearnSequence"),
        "filename_suffix": "_Epochs.fif",
        "process_labels": False,
        "reshape_data": False,
        "resample_freq": None,
        "filter_params": {
            "l_freq": 1,
            "h_freq": 30,
            "filter_kwargs": {"method": "fir", "verbose": False},
            "decimate": 10,
        },
    },
    "preresting": {
        "load": True,
        "process_labels": False,
        "reshape_data": False,
        "resample_freq": None,
        "filter_params": {
            "l_freq": 1,
            "h_freq": 30,
            "filter_kwargs": {"method": "fir", "verbose": False},
            "decimate": 10,
        },
        "merge": [
            {
                "path": Path("Segments", "PreResting"),
                "filename_suffix": "_Epochs.fif",
            },
            {
                "path": Path("Segments", "PreResting"),
                "filename_suffix": "_Break_Epochs.fif",
            },
        ],
    },
}

importer = Things_Importer(INFO_DIR, PREPROC_DIR)

# Import older children (10-13 Months) data segments
participant_info_old = importer.get_participant_info(
    participant_overview,
    participant_filters={"min_age_months": 9, "max_age_months": 14},
)
eeg_data_old = importer.get_eeg_data(
    segments_config=load_segments,
    sample_mask=participant_info_old["participants_incl"],
    verbose=False,
)
behavioral_data_old = importer.get_behavioral_data(eeg_data_old, participant_info_old, RAW_DIR)


# eeg_data arrays are pre-pruned: index 0..n_subs-1 = included participants
n_subs_old = len(participant_info_old["participants_incl"])
sample_mask_old = np.arange(n_subs_old)

# Compute Cohen's kappa for each participant between the two coders
participant_info_old["additional_data"]["localizer"]["cohkappa_scores"] = []
trial_ratings = participant_info_old["additional_data"]["localizer"]["trial_ratings"]
for name in participant_info_old["participants_incl"]:
    idx = trial_ratings["Participant"] == name
    rater1 = trial_ratings.loc[idx, "Attends_Bool_rater1"]
    rater2 = trial_ratings.loc[idx, "Attends_Bool_rater2"]
    kappa = cohen_kappa_score(rater1, rater2)
    participant_info_old["additional_data"]["localizer"]["cohkappa_scores"].append(kappa)


# Import younger children (6-9 Months) data segments
participant_info_young = importer.get_participant_info(
    participant_overview, participant_filters={"min_age_months": 5, "max_age_months": 10}
)
eeg_data_young = importer.get_eeg_data(
    segments_config=load_segments,
    sample_mask=participant_info_young["participants_incl"],
    verbose=False,
)
behavioral_data_young = importer.get_behavioral_data(
    eeg_data_young, participant_info_young, RAW_DIR
)

# eeg_data arrays are pre-pruned: index 0..n_subs-1 = included participants
n_subs_young = len(participant_info_young["participants_incl"])
sample_mask_young = np.arange(n_subs_young)

# Compute Cohen's kappa for each participant between the two coders
participant_info_young["additional_data"]["localizer"]["cohkappa_scores"] = []
trial_ratings = participant_info_young["additional_data"]["localizer"]["trial_ratings"]
for name in participant_info_young["participants_incl"]:
    idx = trial_ratings["Participant"] == name
    rater1 = trial_ratings.loc[idx, "Attends_Bool_rater1"]
    rater2 = trial_ratings.loc[idx, "Attends_Bool_rater2"]
    kappa = cohen_kappa_score(rater1, rater2)
    participant_info_young["additional_data"]["localizer"]["cohkappa_scores"].append(kappa)


# ==============================================================================
# PARAMETERS
# ==============================================================================

# Maximum lag in time points (1 tp = 10 ms at 100 Hz → 50 tp = 500 ms)
MAX_LAG = 50
lags_ms = np.arange(1, MAX_LAG + 1) * 10

# Whether to include the baseline class (index 0) in the analysis.
USE_BASELINE = False

if USE_BASELINE:
    # 4-class forward chain:  baseline(0) → Apple(1) → Chair(2) → Face(3)
    tm, _ = create_transition_matrix([0, 1, 2, 3], [[1], [2], [3], []])
    n_classes = 4
    prob_class_slice = slice(None)
else:
    # 3-class forward chain:  Apple(0) → Chair(1) → Face(2)
    tm, _ = create_transition_matrix([0, 1, 2], [[1], [2], []])
    n_classes = 3
    prob_class_slice = slice(1, None)

CV = 6
solver = "liblinear"
penalty = "l1"
max_iter = 10000

N_SIGN_PERMS = 1000
CLUST_THRESH_PVAL = 0.05
MAX_PVAL = 0.05

FMIN_PSD, FMAX_PSD = 2.0, 25.0
FMIN_BAND, FMAX_BAND = 4.0, 8.0
N_FFT_OSC = 256
OUTLIER_THRESH_MAD = 5.0

# Shared params passed to all pipeline functions
params_shared = {
    "MAX_LAG": MAX_LAG,
    "tm": tm,
    "N_SIGN_PERMS": N_SIGN_PERMS,
    "CLUST_THRESH_PVAL": CLUST_THRESH_PVAL,
    "MAX_PVAL": MAX_PVAL,
    "solver": solver,
    "penalty": penalty,
    "max_iter": max_iter,
    "CV": CV,
    "n_classes": n_classes,
    "FMIN_PSD": FMIN_PSD,
    "FMAX_PSD": FMAX_PSD,
    "FMIN_BAND": FMIN_BAND,
    "FMAX_BAND": FMAX_BAND,
    "N_FFT_OSC": N_FFT_OSC,
    "OUTLIER_THRESH_MAD": OUTLIER_THRESH_MAD,
}

# Group-specific params (differ in Cs and training_timepoint)
params_old = {
    **params_shared,
    "Cs": [6] * n_subs_old,
    "training_timepoint": 0.4,
}
params_young = {
    **params_shared,
    "Cs": [6] * n_subs_young,
    "training_timepoint": 0.45,
}


# ==============================================================================
# LOCALIZER DECODING
# ==============================================================================

classifier_data_old = run_decode(eeg_data_old, sample_mask_old, params_old)
classifier_data_young = run_decode(eeg_data_young, sample_mask_young, params_young)


# ==============================================================================
# CROSS-DECODING
# ==============================================================================

decode_data_old = run_xdecode(eeg_data_old, sample_mask_old, params_old, SEGMENTS)
decode_data_young = run_xdecode(eeg_data_young, sample_mask_young, params_young, SEGMENTS)


# ==============================================================================
# CROSS-CORRELATION + PERMUTATION TESTS
# ==============================================================================

xcorr_data_old = run_xcorr(decode_data_old, params_shared, SEGMENTS)
xcorr_data_young = run_xcorr(decode_data_young, params_shared, SEGMENTS)

# Between-group comparison
xcorr_data_diff = run_group_comparison(xcorr_data_old, xcorr_data_young, params_shared, SEGMENTS)

# Within-group contrasts: cued_replay vs seq_learn, cued_replay vs preresting
xcorr_contr_old = run_contrasts(
    xcorr_data_old, params_shared, "cued_replay", ["seq_learn", "preresting"]
)


# ==============================================================================
# FREQUENCY ANALYSIS
# ==============================================================================

freq_data_old = run_freq_analysis(eeg_data_old, params_shared)
freq_data_young = run_freq_analysis(eeg_data_young, params_shared)
