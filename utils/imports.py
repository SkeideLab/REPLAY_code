# -*- coding: utf-8 -*-
"""
Import Helper Functions.

@author: Christopher Postzich
@github: mcpost
"""


import os
import mne
import numpy as np


def import_preproc_data_replay_things(base_dir, preproc_dir, 
                                      participant_overview,
                                      segments_config=None,
                                      participant_filters=None,
                                      labels=['Apple', 'Chair', 'Face']):
    """
    Import and preprocess EEG data from the Replay Things experiment.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing the experiment data
    preproc_dir : str
        Directory containing preprocessed EEG data segments
    participant_overview : pd.DataFrame
        Excel file containing participant information as DataFrame
    segments_config : dict, optional
        Configuration for which segments to load and how to process them.
        If None, uses default configuration loading all segments.
        Each key is a segment name, with dict containing:
        - 'load': bool, whether to load this segment
        - 'path': str, relative path within preproc_dir
        - 'filename_suffix': str, suffix for epoch files
        - 'process_labels': bool, whether to process stimulus labels
        - 'reshape_data': bool, whether to reshape for continuous data
        - 'resample_freq': int, frequency to resample to (Hz)
    participant_filters : dict, optional
        Filters for participant selection. If None, uses default filters.
        Dictionary with keys:
        - 'baby_only': bool, default=True, filter for baby participants
        - 'include_only': bool, default=True, filter for included participants
        - 'min_Age_Months': int, default=9, minimum age in months
        - 'require_localizer': bool, default=True, require usable localizer data
        - 'require_sequence': bool, default=True, require usable sequence data
        - 'custom_filter': function, optional, custom filter function that takes participant_baby df
    labels : list, default=['Apple', 'Chair', 'Face']
        Labels for stimulus categories
    
    Returns:
    --------
    tuple of (participant_info, eeg_data, behavioral_data)
        participant_info : dict
            Dictionary containing participant information:
            - 'participant_baby': Baby participants subset
            - 'participant_included': Final included participants
            - 'pilotnames_all': All pilot names
            - 'pilotnames_incl': Included pilot names
            - 'convidx_all_incl': Conversion indices
        eeg_data : dict
            Dictionary containing EEG data for each segment type with preprocessing info
        behavioral_data : dict
            Dictionary containing behavioral summary and demographics:
            - 'behavioral_summary': Basic behavioral summary stats
            - 'labels': Stimulus labels
    
    Example usage:
    --------------
        from utils.import import import_preproc_data_replay_things
        
        # Custom participant filtering
        custom_participant_filters = {
            'baby_only': True,
            'include_only': True,
            'min_Age_Months': 12,  # Only participants older than 12 months
            'require_localizer': True,
            'require_sequence': False,  # Don't require sequence data
            'custom_filter': lambda df: df[df['Gender'] == 'M']  # Only male participants
        }
        
        # Custom segments config - only localizer with high sampling
        custom_segments = {
            'localizer': {
                'load': True,
                'path': os.path.join('Segments', 'Localizer'),
                'filename_suffix': '_Epochs.fif',
                'process_labels': True,
                'reshape_data': False,
                'resample_freq': 250
            }
        }
        
        # Function now returns three separate dictionaries
        participant_info, eeg_data, behavioral_data = import_preproc_data_replay_things(
            base_dir='/path/to/base',
            preproc_dir='/path/to/preproc',
            segments_config=custom_segments,
            participant_filters=custom_participant_filters
        )
        
        # Access specific data from each output
        included_participants = participant_info['participant_included']
        localizer_epochs = eeg_data['localizer']['epochs']
        preprocessing_info = eeg_data['localizer']['preprocessing_info']
        trial_counts = behavioral_data['behavioral_summary']['trialnum_loc']
        stimulus_labels = behavioral_data['labels']
    """
    
    # Define participant filters
    if participant_filters is None:
        participant_filters = {
            'baby_only': True,
            'include_only': True,
            'min_Age_Months': 9,
            'require_localizer': True,
            'require_sequence': True,
            'custom_filter': None
        }
    
    # Start with all participants or those with Include data
    if participant_filters.get('include_only', True):
        temp = participant_overview.dropna(subset='Include')
    else:
        temp = participant_overview.copy()
    
    # Filter for baby participants if requested
    if participant_filters.get('baby_only', True):
        participant_baby = temp[temp.Baby.astype(bool)]
    else:
        participant_baby = temp.copy()
    
    
    # Define pilotnames
    pilotnames_all = participant_baby.Participant.to_list()
    
    # Initialize data containers
    eeg_data = {}
    
    # Define segment configurations
    if segments_config is None:
        segments_config = {
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
                'reshape_data': True,
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
            'pre_resting': {
                'load': True,
                'path': os.path.join('Segments', 'PreResting'),
                'filename_suffix': '_Epochs.fif',
                'process_labels': False,
                'reshape_data': True,
                'resample_freq': 100
            },
            'pre_resting_break': {
                'load': True,
                'path': os.path.join('Segments', 'PreResting'),
                'filename_suffix': '_Break_Epochs.fif',
                'process_labels': False,
                'reshape_data': True,
                'resample_freq': 100
            }
        }
    
    # Load data for each segment type
    for segment_name, config in segments_config.items():
        if not config['load']:
            continue
            
        # Initialize arrays for this segment
        epochs_array = np.empty((len(pilotnames_all),), dtype=object)
        data_array = np.empty((len(pilotnames_all),), dtype=object)
        labels_array = np.empty((len(pilotnames_all),), dtype=object)
        
        for p in range(len(participant_baby)):
            file_path = os.path.join(
                preproc_dir, config['path'], 
                f"{pilotnames_all[p]}{config['filename_suffix']}"
            )
            
            if os.path.exists(file_path):
                epochs = mne.read_epochs(file_path, preload=True)
                epochs.resample(config['resample_freq'])
                
                epochs_array[p] = epochs
                
                # Process data based on segment type
                if config['reshape_data']:
                    # For resting state data - reshape to continuous
                    data_array[p] = np.reshape(
                        np.transpose(epochs.copy().get_data(copy=False), (1,0,2)),
                        (len(epochs.ch_names), -1)
                    )
                    labels_array[p] = np.ones(len(epochs))
                else:
                    # For trial-based data
                    data_array[p] = epochs.copy().get_data(copy=False)
                    
                    if config['process_labels'] and segment_name == 'localizer':
                        # Process localizer labels
                        trl_index_loc = np.vstack([
                            [1 if 'apple' in a else 0 for a in epochs.metadata['Loc_Image']],
                            [2 if 'chair' in a else 0 for a in epochs.metadata['Loc_Image']],
                            [3 if 'face' in a else 0 for a in epochs.metadata['Loc_Image']]
                        ])
                        labels_array[p] = trl_index_loc.sum(0)
                    else:
                        labels_array[p] = np.ones(len(epochs))
        
        # Store in eeg_data dict with preprocessing info
        eeg_data[segment_name] = {
            'epochs': epochs_array,
            'data': data_array,
            'trial_labels': labels_array,
            'times': epochs_array[0].times,
            'ch_names': epochs_array[0].ch_names,
            'info': epochs_array[0].info,
            'preprocessing': {
                'resample_freq': config['resample_freq'],
                'process_labels': config['process_labels'],
                'reshape_data': config['reshape_data'],
                'path': config['path'],
                'filename_suffix': config['filename_suffix']
            }
        }
    
    # Define Subset of included Participants based on filters
    filters = []
    
    # Always start with non-null participants if include_only is True
    if participant_filters.get('include_only', True):
        filters.append(participant_baby.Include.astype(bool))
    
    # Add age filter
    min_age = participant_filters.get('min_Age_Months', 9)
    filters.append(participant_baby.Age_Months > min_age)
    
    # Add localizer requirement
    if participant_filters.get('require_localizer', True):
        filters.append(participant_baby.Localizer_usable.astype(bool))
    
    # Add sequence requirement
    if participant_filters.get('require_sequence', True):
        filters.append(participant_baby.Sequence_usable.astype(bool))
    
    # Combine all filters
    if filters:
        combined_filter = filters[0]
        for f in filters[1:]:
            combined_filter = combined_filter & f
        participant_included = participant_baby[combined_filter]
    else:
        participant_included = participant_baby.copy()
    
    # Apply custom filter if provided
    if participant_filters.get('custom_filter') is not None:
        participant_included = participant_filters['custom_filter'](participant_included)
    
    # Define pilotnames for included participants
    pilotnames_incl = participant_included.Participant.to_list()
    
    # Define conversion indexes between "all" and "included"
    # Recreate the same filter logic to get indices
    filters_for_idx = []
    if participant_filters.get('include_only', True):
        filters_for_idx.append(participant_baby.Include.astype(bool))
    
    min_age = participant_filters.get('min_Age_Months', 9)
    filters_for_idx.append(participant_baby.Age_Months > min_age)
    
    if participant_filters.get('require_localizer', True):
        filters_for_idx.append(participant_baby.Localizer_usable.astype(bool))
    
    if participant_filters.get('require_sequence', True):
        filters_for_idx.append(participant_baby.Sequence_usable.astype(bool))
    
    if filters_for_idx:
        combined_filter_idx = filters_for_idx[0]
        for f in filters_for_idx[1:]:
            combined_filter_idx = combined_filter_idx & f
        convidx_all_incl = np.where(combined_filter_idx)[0]
    else:
        convidx_all_incl = np.arange(len(participant_baby))
    
    # Generate behavioral summary if relevant data was loaded
    behavioral_summary = {}
    if 'localizer' in segments_config and segments_config['localizer']['load'] and 'localizer' in eeg_data:
        behavioral_summary['trialnum_loc'] = np.array([
            [np.sum(lab == l) for l in [1,2,3]] 
            for lab in eeg_data['localizer']['trial_labels'][convidx_all_incl]
            if lab is not None
        ])
    
    if 'resting' in segments_config and segments_config['resting']['load'] and 'resting' in eeg_data:
        behavioral_summary['time_rest'] = np.array([
            dat.shape[-1]/segments_config['resting']['resample_freq'] if dat is not None and len(dat) else 0.0 
            for dat in eeg_data['resting']['data'][convidx_all_incl]
        ])
    
    # Get demographics for included participants
    behavioral_summary['pilotdemog'] = (
        participant_included['Age_Months'].astype(str) + 
        participant_included['Gender'].astype(str)
    ).to_numpy()
    
    # Compile results into three separate dictionaries
    participant_info = {
        'participant_baby': participant_baby,
        'participant_included': participant_included,
        'pilotnames_all': pilotnames_all,
        'pilotnames_incl': pilotnames_incl,
        'convidx_all_incl': convidx_all_incl
    }
    
    behavioral_data = {
        'behavioral_summary': behavioral_summary,
        'labels': labels
    }
    
    return participant_info, eeg_data, behavioral_data



