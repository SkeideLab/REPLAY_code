# -*- coding: utf-8 -*-
"""
Load up and Transform data segments.

@author: Christopher Postzich
@github: mcpost
"""

# %% Imports

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from mne.baseline import rescale
from mne.decoding import (
    LinearModel,
    SlidingEstimator,
    cross_val_multiscore,
    get_coef,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from utils.TDLM import TDLM, create_transition_matrix
from utils.CrossDecoding_MEEG import CrossDecoding_MEEG
from utils.imports import import_preproc_data_replay_things
from utils.utils import sign_flip_permtest


# %% Define necessary paths

code_dir = os.path.join('')
raw_dir = os.path.join("X:\\","REPLAY","raw","Things") #os.path.join('')
preproc_dir = os.path.join("X:\\","REPLAY","preprocessed","Things") #os.path.join('')


# %% Parameters

labels = ['Apple','Chair','Face']


# %% Load Data

# Which data segments to transform
transform_segments = [
    'resting',
    'cued_replay',
    'seq_learn',
    'preresting',
    'preresting_break'
]

# Import Participant Information
participant_overview = pd.read_excel(
    os.path.join(code_dir, 'additional_data/participant_info.xlsx'),
    engine='calamine', decimal=','
)

# Which data segments to import
load_segments = {
    'localizer': {
        'load': True,
        'path': os.path.join('Segments', 'Localizer'),
        'filename_suffix': '_Epochs.fif',
        'process_labels': True,
        'reshape_data': False,
        'resample_freq': 100
    },
    'resting': {
        'load': True,
        'path': os.path.join('Segments', 'Resting'),
        'filename_suffix': '_Epochs.fif',
        'process_labels': False,
        'reshape_data': False,
        'resample_freq': 100
    },
    'cued_replay': {
        'load': True,
        'path': os.path.join('Segments', 'CuedReplay'),
        'filename_suffix': '_Epochs.fif',
        'process_labels': False,
        'reshape_data': False,
        'resample_freq': 100
    },
    'seq_learn': {
        'load': True,
        'path': os.path.join('Segments', 'LearnSequence'),
        'filename_suffix': '_Epochs.fif',
        'process_labels': False,
        'reshape_data': False,
        'resample_freq': 100
    },
    'preresting': {
        'load': True,
        'path': os.path.join('Segments', 'PreResting'),
        'filename_suffix': '_Epochs.fif',
        'process_labels': False,
        'reshape_data': False,
        'resample_freq': 100
    },
    'preresting_break': {
        'load': True,
        'path': os.path.join('Segments', 'PreResting'),
        'filename_suffix': '_Break_Epochs.fif',
        'process_labels': False,
        'reshape_data': False,
        'resample_freq': 100
    },
}

# Import data segments
participant_info, eeg_data, _ = import_preproc_data_replay_things(
    code_dir,
    preproc_dir,
    participant_overview,
    participant_filters={'baby_only': False, 'include_only': False,
                         'min_age_months': 9, 'require_localizer': False,
                         'require_sequence': False},
    segments_config = load_segments,
)

# Sample mask
sample_mask = participant_info['convidx_all_incl']


# %% Behavioral Measures and Demographics

# Rater Agreement
participant_info['localizer_ratings'] = pd.DataFrame()
participant_info['localizer_cohkappa'] = np.zeros(len(sample_mask))

participant_info['loc_trialnum'] = np.zeros((len(sample_mask), 3))
participant_info['time_rest'] = np.zeros((len(sample_mask), 1))
for p,idx in enumerate(tqdm(sample_mask, desc="Subject Ratings loaded")):

    # Number of localizer trials
    _, participant_info['loc_trialnum'][p,:] = np.unique(
        eeg_data['localizer']['trial_labels'][idx],
        return_counts=True
    )

    # Amount of Resting time
    participant_info['time_rest'][p,:] = np.prod(
        eeg_data['resting']['data'][idx].shape[0:3:2]
    )/100

    rater1 = pd.read_csv(os.path.join(
        raw_dir,
        'ratings',
        'rater1',
        f'Template_Localizer_{participant_info["pilotnames_incl"][p]}.csv'),
        sep=';'
    ).dropna(subset='Block')
    rater1.insert(0,'Participant',participant_info["pilotnames_incl"][p])

    rater2 = pd.read_excel(os.path.join(
        raw_dir,
        'ratings',
        'rater2',
        f'Template_Localizer_{participant_info["pilotnames_incl"][p]}.xlsx'),
        engine='calamine',
        decimal=','
    ).dropna(subset='Block')
    rater2.insert(0,'Participant',participant_info["pilotnames_incl"][p])

    participant_info['localizer_cohkappa'][p] = cohen_kappa_score(
        rater1['Attends_Bool'],
        rater2['Attends_Bool']
    )

    participant_info['localizer_ratings'] = pd.concat([
        participant_info['localizer_ratings'],
        pd.merge(rater1,
        rater2,
        on=['Participant', 'Block', 'Trials'],
        suffixes=("_rater1","_rater2"))
    ])

del rater1, rater2, p, idx


# Get demographics names
participant_info['info_names'] = (
    participant_info['participant_included']['Age_Months'].astype(str)
    + participant_info['participant_included']['Gender'].astype(str)
).to_numpy()

# Age
participant_info['mean_age'] = participant_info['participant_included']['Age_Months'].mean()

# Trial Rejection
participant_info['mean_trlrej_loc'] = np.mean(
    1-(participant_info['loc_trialnum'].sum(1)/540)
)
participant_info['std_trlrej_loc'] = np.std(
    1-(participant_info['loc_trialnum'].sum(1)/540)
)

# Resting Time
participant_info['mean_time_rest'] = np.mean(participant_info['time_rest'])
participant_info['std_time_rest'] = np.std(participant_info['time_rest'])


# %% Classifier Analysis on Localizer Data

# Logisitic Regression Parameters
CV = 6
C = 6
solver = 'liblinear'
penalty = 'l1'
max_iter = 10000


classifier_data = {
    'clf': [],
    'scoring': "roc_auc_ovr",
    'spatial_patterns': [[]]*len(sample_mask),
    'performance': np.ndarray(
        (len(sample_mask), CV, len(eeg_data['localizer']['times']))
        ),
    'permutations': {
        'cluster_dict': {},
        'n_perms': 1000,
        'cluster_form_pval': 0.01,
        'cluster_max_pval': 0.01,
        'statistic': 'cluster_tsum',
    },
}


# Logistic Regression: Define Machine Learning Pipeline
classifier_data['clf'] = make_pipeline(
    StandardScaler(),
    LinearModel(LogisticRegression(
        C=C,
        solver=solver,
        penalty=penalty,
        max_iter=max_iter
        )
    )
)

for p,data,label in tqdm(zip(
            range(len(sample_mask)),
            eeg_data['localizer']['data'][sample_mask],
            eeg_data['localizer']['trial_labels'][sample_mask]),
        total=len(sample_mask),
        desc='Participants cross-validated'):

    if data is not None:
        # Apply Baseline
        data_bl = rescale(
            data,
            eeg_data['localizer']['times'],
            (-0.2, 0.0),
            'mean',
            verbose=False
        )

        ## Overall Classification: One vs Rest
        # Define Sliding Estimator
        time_decod = SlidingEstimator(
            classifier_data['clf'],
            n_jobs=None,
            scoring=classifier_data['scoring'],
            verbose=False
        )
        # Fit across time points
        time_decod.fit(data_bl, label)
        # Get Spatial Patterns
        classifier_data['spatial_patterns'][p] = get_coef(
            time_decod,
            "patterns_",
            inverse_transform=True
        )
        # Cross-validated Training (10 Folds)
        classifier_data['performance'][p,:,:] = cross_val_multiscore(
            time_decod,
            data_bl,
            label,
            cv=CV,
            verbose=False
        )

del data, label, p

# Quick Permutation Test
classifier_data['permutations']['cluster_dict'] = sign_flip_permtest(
    classifier_data['performance'],
    classifier_data['permutations']['n_perms'],
    classifier_data['permutations']['cluster_form_pval'],
    chance_lev = 0.5
)


# %% Decode Data Segments

decode_data = {}
# Instantiate a classifier pipeline
decode_data['clf'] = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        C=C,
        solver=solver,
        penalty=penalty,
        max_iter=max_iter
    ))
])

for name in transform_segments:

    decode_data[name] = {
        'decoder': [],
        'probabilities': [[]]*len(sample_mask),
        'decision_values': [[]]*len(sample_mask),
    }


    # Create cross-decoder with specific training time and baseline
    decode_data[name]['decoder'] = CrossDecoding_MEEG(
        base_estimator=decode_data['clf'],
        train_times=eeg_data['localizer']['times'],
        test_times=eeg_data[name]['times'],
        training_timepoint=0.4,
        baseline_timepoint=-0.1,
        include_zero=True,
        random_state=42,
        verbose=True
    )


    for p,x_train,y_train,x_test in zip(range(len(sample_mask)),
                                        eeg_data['localizer']['data'][sample_mask],
                                        eeg_data['localizer']['trial_labels'][sample_mask],
                                        eeg_data[name]['data'][sample_mask]):

        if x_test is not None:
            # Create and fit decoder
            decode_data[name]['decoder'].fit(x_train, y_train)

            # Make predictions
            decode_data[name]['probabilities'][p] = decode_data[name]['decoder'].predict_proba(x_test)

            decode_data[name]['decision_values'][p] = decode_data[name]['decoder'].decision_function(x_test)

    del x_train, y_train, x_test, p

    #decode_data[name]['probabilities'] = np.array(probabilities)
    #decode_data[name]['decision_values'] = np.array(decision_values)


# %% Run TDLM on Data Segments

tdlm_data = {}
# Instantiate a TDLM class
tdlm_data['max_lag'] = 50
tdlm_data['bin_lag'] = 8
tdlm_data['uperm_method'] = 'uperms'
tdlm_data['tdlm'] = TDLM(
    tdlm_data['max_lag'],
    tdlm_data['bin_lag'],
    tdlm_data['uperm_method']
)
tdlm_data['classes'] = decode_data[name]['decoder'].classes_
# Theoretical Transition Matrix
# Forward Direction:  0->1, 1->2, 2->3
# Backward Direction: 1->0, 2->1, 3->2
tdlm_data['tm'], _ = create_transition_matrix([0,1,2,3], [[1],[2],[3],[]])
# Compute Permutation of TTM
tm_perms = tdlm_data['tdlm']._get_tm_permutations(model_tm=tdlm_data['tm'])


for name in tqdm(transform_segments, desc='Segments transformed'):

    tdlm_data[name] = {
        'sequenceness': [],
        'permutations': [],
        'empirical_tm': [],
        'replay_onsets': [],
        'num_timepoints': []
    }

    # Apply TDLM to probability data
    tdlm_data[name]['sequenceness'] = np.nan*np.ones(
        (len(sample_mask), 4, tdlm_data['max_lag'])
    )
    tdlm_data[name]['permutations'] = np.nan*np.ones(
        (len(sample_mask), len(tm_perms), 4, tdlm_data['max_lag'])
    )
    for p,prob in zip(range(len(sample_mask)),
                      decode_data[name]['probabilities']):

        if np.any(prob):
            # Reshape Data
            cur_prob = prob.transpose(1, 0, 2).reshape(4, -1)

            # Get Sequenceness Measure over Lags
            tdlm_data[name]['sequenceness'][p,:,:] = tdlm_data['tdlm'].fit(
                cur_prob,
                tdlm_data['classes'],
                model_tm=tdlm_data['tm']
            )

            # Get Permutations of Sequenceness Measure over Lags
            tdlm_data[name]['permutations'][p,:,:,:] = tdlm_data['tdlm'].permutations(
                cur_prob,
                tdlm_data['classes'],
                model_tm_perms=tm_perms
            )

            tdlm_data[name]['empirical_tm'] = tdlm_data['tdlm'].empirical_tm

            tdlm_data[name]['replay_onsets'].append(
                tdlm_data['tdlm'].replay_onsets(
                    cur_prob,
                    lag=5,
                    model_tm=tdlm_data['tm']
                )
            )

            tdlm_data[name]['num_timepoints'].append(
                prob.shape[0]*prob.shape[2]
            )

del prob, p, cur_prob, tm_perms


# %% Run TDLM on Data Segments (No Binning)

tdlm_data_non = {}
# Instantiate a TDLM class
tdlm_data_non['max_lag'] = 50
tdlm_data_non['bin_lag'] = None
tdlm_data_non['uperm_method'] = 'uperms'
tdlm_data_non['tdlm'] = TDLM(
    tdlm_data_non['max_lag'],
    tdlm_data_non['bin_lag'],
    tdlm_data_non['uperm_method']
)
tdlm_data_non['classes'] = decode_data[name]['decoder'].classes_
# Theoretical Transition Matrix
# Forward Direction:  0->1, 1->2, 2->3
# Backward Direction: 1->0, 2->1, 3->2
tdlm_data_non['tm'], _ = create_transition_matrix([0,1,2,3], [[1],[2],[3],[]])
# Compute Permutation of TTM
tm_perms = tdlm_data_non['tdlm']._get_tm_permutations(model_tm=tdlm_data_non['tm'])


for name in tqdm(transform_segments, desc='Segments transformed'):

    tdlm_data_non[name] = {
        'sequenceness': [],
        'permutations': [],
        'empirical_tm': [],
        'replay_onsets': [],
        'num_timepoints': []
    }

    # Apply TDLM to probability data
    tdlm_data_non[name]['sequenceness'] = np.nan*np.ones(
        (len(sample_mask), 4, tdlm_data_non['max_lag'])
    )
    tdlm_data_non[name]['permutations'] = np.nan*np.ones(
        (len(sample_mask), len(tm_perms), 4, tdlm_data_non['max_lag'])
    )
    for p,prob in zip(range(len(sample_mask)),
                      decode_data[name]['probabilities']):

        if np.any(prob):
            # Reshape Data
            cur_prob = prob.transpose(1, 0, 2).reshape(4, -1)

            # Get Sequenceness Measure over Lags
            tdlm_data_non[name]['sequenceness'][p,:,:] = tdlm_data_non['tdlm'].fit(
                cur_prob,
                tdlm_data_non['classes'],
                model_tm=tdlm_data_non['tm']
            )

            # Get Permutations of Sequenceness Measure over Lags
            tdlm_data_non[name]['permutations'][p,:,:,:] = tdlm_data_non['tdlm'].permutations(
                cur_prob,
                tdlm_data_non['classes'],
                model_tm_perms=tm_perms
            )

            tdlm_data_non[name]['empirical_tm'] = tdlm_data_non['tdlm'].empirical_tm

            tdlm_data_non[name]['replay_onsets'].append(
                tdlm_data_non['tdlm'].replay_onsets(
                    cur_prob,
                    lag=5,
                    model_tm=tdlm_data_non['tm']
                )
            )

            tdlm_data_non[name]['num_timepoints'].append(
                prob.shape[0]*prob.shape[2]
            )

del prob, p, cur_prob, tm_perms


# %% Run TDLM on Data Segments (Full Transition Matrix)

tdlm_data_adv = {}
# Instantiate a TDLM class
tdlm_data_adv['max_lag'] = 50
tdlm_data_adv['bin_lag'] = 8
tdlm_data_adv['uperm_method'] = 'uperms'
tdlm_data_adv['tdlm'] = TDLM(
    tdlm_data_adv['max_lag'],
    tdlm_data_adv['bin_lag'],
    tdlm_data_adv['uperm_method']
)
tdlm_data_adv['classes'] = decode_data[name]['decoder'].classes_
# Theoretical Transition Matrix
# Forward Direction:  0->1, 1->2, 2->3
# Backward Direction: 1->0, 2->1, 3->2
tdlm_data_adv['tm'], _ = create_transition_matrix([0,1,2,3], [[1],[2],[3],[0]])
# Compute Permutation of TTM
tm_perms = tdlm_data_adv['tdlm']._get_tm_permutations(model_tm=tdlm_data_adv['tm'])


for name in tqdm(transform_segments, desc='Segments transformed'):

    tdlm_data_adv[name] = {
        'sequenceness': [],
        'permutations': [],
        'empirical_tm': [],
        'replay_onsets': [],
        'num_timepoints': []
    }

    # Apply TDLM to probability data
    tdlm_data_adv[name]['sequenceness'] = np.nan*np.ones(
        (len(sample_mask), 4, tdlm_data_adv['max_lag'])
    )
    tdlm_data_adv[name]['permutations'] = np.nan*np.ones(
        (len(sample_mask), len(tm_perms), 4, tdlm_data_adv['max_lag'])
    )
    for p,prob in zip(range(len(sample_mask)),
                      decode_data[name]['probabilities']):

        if np.any(prob):
            # Reshape Data
            cur_prob = prob.transpose(1, 0, 2).reshape(4, -1)

            # Get Sequenceness Measure over Lags
            tdlm_data_adv[name]['sequenceness'][p,:,:] = tdlm_data_adv['tdlm'].fit(
                cur_prob,
                tdlm_data_adv['classes'],
                model_tm=tdlm_data_adv['tm']
            )

            # Get Permutations of Sequenceness Measure over Lags
            tdlm_data_adv[name]['permutations'][p,:,:,:] = tdlm_data_adv['tdlm'].permutations(
                cur_prob,
                tdlm_data_adv['classes'],
                model_tm_perms=tm_perms
            )

            tdlm_data_adv[name]['empirical_tm'] = tdlm_data_adv['tdlm'].empirical_tm

            tdlm_data_adv[name]['replay_onsets'].append(
                tdlm_data_adv['tdlm'].replay_onsets(
                    cur_prob,
                    lag=5,
                    model_tm=tdlm_data_adv['tm']
                )
            )

            tdlm_data_adv[name]['num_timepoints'].append(
                prob.shape[0]*prob.shape[2]
            )

del prob, p, cur_prob, tm_perms


# %% Run TDLM on Data Segments (No Baseline)

tdlm_data_nobl = {}
# Instantiate a TDLM class
tdlm_data_nobl['max_lag'] = 50
tdlm_data_nobl['bin_lag'] = 8
tdlm_data_nobl['uperm_method'] = 'uperms'
tdlm_data_nobl['tdlm'] = TDLM(
    tdlm_data_nobl['max_lag'],
    tdlm_data_nobl['bin_lag'],
    tdlm_data_nobl['uperm_method']
)
tdlm_data_nobl['classes'] = decode_data[name]['decoder'].classes_[:3]
# Theoretical Transition Matrix
# Forward Direction:  0->1, 1->2, 2->3
# Backward Direction: 1->0, 2->1, 3->2
tdlm_data_nobl['tm'], _ = create_transition_matrix([0,1,2], [[1],[2],[]])
# Compute Permutation of TTM
tm_perms = tdlm_data_nobl['tdlm']._get_tm_permutations(model_tm=tdlm_data_nobl['tm'])

for name in tqdm(transform_segments, desc='Segments transformed'):

    tdlm_data_nobl[name] = {
        'sequenceness': [],
        'permutations': [],
        'empirical_tm': [],
        'replay_onsets': [],
        'num_timepoints': []
    }

    # Apply TDLM to probability data
    tdlm_data_nobl[name]['sequenceness'] = np.nan*np.ones(
        (len(sample_mask), 4, tdlm_data_nobl['max_lag'])
    )
    tdlm_data_nobl[name]['permutations'] = np.nan*np.ones(
        (len(sample_mask), len(tm_perms), 4, tdlm_data_nobl['max_lag'])
    )
    for p,prob in zip(range(len(sample_mask)),
                      decode_data[name]['probabilities']):

        if np.any(prob):
            # Reshape Data
            cur_prob = prob[:,1:,:].transpose(1, 0, 2).reshape(3, -1)

            # Get Sequenceness Measure over Lags
            tdlm_data_nobl[name]['sequenceness'][p,:,:] = tdlm_data_nobl['tdlm'].fit(
                cur_prob,
                tdlm_data_nobl['classes'],
                model_tm=tdlm_data_nobl['tm']
            )

            # Get Permutations of Sequenceness Measure over Lags
            tdlm_data_nobl[name]['permutations'][p,:,:,:] = tdlm_data_nobl['tdlm'].permutations(
                cur_prob,
                tdlm_data_nobl['classes'],
                model_tm_perms=tm_perms
            )

            tdlm_data_nobl[name]['empirical_tm'] = tdlm_data_nobl['tdlm'].empirical_tm

            tdlm_data_nobl[name]['replay_onsets'].append(
                tdlm_data_nobl['tdlm'].replay_onsets(
                    cur_prob,
                    lag=5,
                    model_tm=tdlm_data_nobl['tm']
                )
            )

            tdlm_data_nobl[name]['num_timepoints'].append(
                prob.shape[0]*prob.shape[2]
            )

del prob, p, cur_prob, tm_perms


# %% Cued Replay

temp = []
for prob in decode_data['cued_replay']['probabilities']:

    if prob.shape[0] >= 10:

        temp.append(
             (rescale(
                prob,
                eeg_data['cued_replay']['times'],
                (-0.1,-0.0),
                "mean"
            )).mean(0,keepdims=True)
        )
cur_data = np.concatenate(temp)
cond_data = [np.concatenate([cur_data[:,l,:]]) for l in range(1,4)]

stim_pval = [[]]*3
with np.load("additional_data/clusterdepth_pvals.npz", allow_pickle=True) as f:
    stim_pval[0] = f['cluster_pval1']
    stim_pval[1] = f['cluster_pval2']
    stim_pval[2] = f['cluster_pval3']

perm_info = {
        'stat_data': np.random.randn((121)),
        'dims': 'times',
        'times': np.linspace(-0.2, 1.0, 121),
        'threshold_pval': 0.05,
        'method': 'cluster_depth',
        'permtest_pval': 0.05,
        'corrected_pvals': [],
        'mask': []  # Another significant cluster
    }

del temp, cur_data


# %% Save Analysis Data (Optional - skip in interactive sessions)
# Set SAVE_DATA = False to skip saving when running interactively

SAVE_DATA = True

if SAVE_DATA:
    import pickle

    # Define output directory
    results_dir = os.path.join(code_dir, '..', 'Results')
    os.makedirs(results_dir, exist_ok=True)

    # Save all analysis data to a single file
    analysis_data = {
        # Metadata
        'labels': labels,
        'transform_segments': transform_segments,
        'sample_mask': sample_mask,

        # Participant info (excluding large dataframes)
        'participant_info': {
            'info_names': participant_info['info_names'],
            'mean_age': participant_info['mean_age'],
            'loc_trialnum': participant_info['loc_trialnum'],
            'time_rest': participant_info['time_rest'],
            'localizer_cohkappa': participant_info['localizer_cohkappa'],
            'mean_trlrej_loc': participant_info['mean_trlrej_loc'],
            'std_trlrej_loc': participant_info['std_trlrej_loc'],
            'mean_time_rest': participant_info['mean_time_rest'],
            'std_time_rest': participant_info['std_time_rest'],
            'participant_included': participant_info['participant_included'],
        },

        # EEG data (times only, not raw data)
        'eeg_times': {seg: eeg_data[seg]['times'] for seg in eeg_data.keys()},
        'eeg_epochs_info': eeg_data['localizer']['epochs'][0].info if eeg_data['localizer']['epochs'] else None,

        # Classifier data
        'classifier_data': {
            'performance': classifier_data['performance'],
            'spatial_patterns': classifier_data['spatial_patterns'],
            'permutations': classifier_data['permutations'],
            'scoring': classifier_data['scoring'],
        },

        # Decode data (probabilities and decision values)
        'decode_data': {
            seg: {
                'probabilities': decode_data[seg]['probabilities'],
                'decision_values': decode_data[seg]['decision_values'],
            } for seg in transform_segments
        },

        # TDLM data
        'tdlm_data': {
            'max_lag': tdlm_data['max_lag'],
            'bin_lag': tdlm_data['bin_lag'],
            'classes': tdlm_data['classes'],
            **{seg: tdlm_data[seg] for seg in transform_segments}
        },
        'tdlm_data_non': {
            'max_lag': tdlm_data_non['max_lag'],
            'bin_lag': tdlm_data_non['bin_lag'],
            **{seg: tdlm_data_non[seg] for seg in transform_segments}
        },
        'tdlm_data_adv': {
            'max_lag': tdlm_data_adv['max_lag'],
            'bin_lag': tdlm_data_adv['bin_lag'],
            **{seg: tdlm_data_adv[seg] for seg in transform_segments}
        },
        'tdlm_data_nobl': {
            'max_lag': tdlm_data_nobl['max_lag'],
            'bin_lag': tdlm_data_nobl['bin_lag'],
            **{seg: tdlm_data_nobl[seg] for seg in transform_segments}
        },

        # Cued replay
        'cond_data': cond_data,
        'stim_pval': stim_pval,
        'perm_info': perm_info,
    }

    # Save as pickle
    output_path = os.path.join(results_dir, 'analysis_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(analysis_data, f)

    print(f"Analysis data saved to: {output_path}")
