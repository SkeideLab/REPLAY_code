# -*- coding: utf-8 -*-
"""
Import Helper Functions

Created on Thu Dec 12 12:54:20 2024
@author: Christopher Postzich
"""

import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_participant_overview(info_dir):
    """Read participant overview and compute Age_Months for every row.

    Parameters
    ----------
    info_dir : str or Path
        Directory containing participants_info.xlsx.

    Returns
    -------
    pd.DataFrame
        Full participant overview with an Age_Months column added.
    """
    overview = pd.read_excel(
        Path(info_dir) / "participants_info.xlsx",
        engine="calamine",
        decimal=",",
    )

    return overview


# =============================================================================
# Things_Importer class
# =============================================================================


class Things_Importer:
    """Import and preprocess EEG data from the Replay Things experiment.

    Parameters
    ----------
    info_dir : str or Path
        Directory containing participants_info.xlsx.
    preproc_dir : str or Path
        Directory containing preprocessed EEG segment folders.
    labels : list, optional
        Stimulus category labels. Default: ["Apple", "Chair", "Face"].
    """

    def __init__(self, info_dir, preproc_dir, labels=None):
        self.info_dir = Path(info_dir)
        self.preproc_dir = Path(preproc_dir)
        self.labels = labels or ["Apple", "Chair", "Face"]
        self._pilotnames_all = None

    # -------------------------------------------------------------------------
    def get_participant_info(self, participant_overview=None, participant_filters=None):
        """Apply inclusion filters to the participant overview.

        Parameters
        ----------
        participant_overview : pd.DataFrame, optional
            Output of load_participant_overview(). If None the overview is
            loaded from self.info_dir automatically (for backward compat).
        participant_filters : dict, optional
            Keys: baby_only (bool), include_only (bool), min_age_months (int),
            max_age_months (int), require_localizer (bool),
            require_sequence (bool), custom_filter (callable or None).

        Returns
        -------
        dict with keys: participant_baby, participant_included, pilotnames_all,
            pilotnames_incl, participants_incl (alias for pilotnames_incl),
            convidx_all_incl.
        """
        if participant_overview is None:
            participant_overview = load_participant_overview(self.info_dir)

        if participant_filters is None:
            participant_filters = {
                "baby_only": True,
                "include_only": True,
                "min_age_months": 9,
                "max_age_months": 14,
                "require_localizer": True,
                "require_sequence": True,
                "custom_filter": None,
            }

        if participant_filters.get("include_only", True):
            temp = participant_overview.dropna(subset="Include")
        else:
            temp = participant_overview.copy()

        if participant_filters.get("baby_only", True):
            participant_baby = temp[temp.Baby.astype(bool)].copy()
        else:
            participant_baby = temp.copy()

        # Age_Months is already present from load_participant_overview

        pilotnames_all = participant_baby.Participant.to_list()
        self._pilotnames_all = pilotnames_all

        # --- included participants ---
        filters = []
        if participant_filters.get("include_only", True):
            filters.append(participant_baby.Include.astype(bool))
        min_age = participant_filters.get("min_age_months", 9)
        filters.append(participant_baby.Age_Months > min_age)
        max_age = participant_filters.get("max_age_months", 14)
        filters.append(participant_baby.Age_Months < max_age)
        if participant_filters.get("require_localizer", True):
            filters.append(participant_baby.Localizer_usable.astype(bool))
        if participant_filters.get("require_sequence", True):
            filters.append(participant_baby.Sequence_usable.astype(bool))

        if filters:
            combined_filter = filters[0]
            for f in filters[1:]:
                combined_filter = combined_filter & f
            participant_included = participant_baby[combined_filter]
        else:
            participant_included = participant_baby.copy()

        if participant_filters.get("custom_filter") is not None:
            participant_included = participant_filters["custom_filter"](participant_included)

        pilotnames_incl = participant_included.Participant.to_list()

        # --- integer index mapping ---
        filters_for_idx = []
        if participant_filters.get("include_only", True):
            filters_for_idx.append(participant_baby.Include.astype(bool))
        filters_for_idx.append(participant_baby.Age_Months > 1)
        if participant_filters.get("require_localizer", True):
            filters_for_idx.append(participant_baby.Localizer_usable.astype(bool))
        if participant_filters.get("require_sequence", True):
            filters_for_idx.append(participant_baby.Sequence_usable.astype(bool))

        if filters_for_idx:
            combined_filter_idx = filters_for_idx[0]
            for f in filters_for_idx[1:]:
                combined_filter_idx = combined_filter_idx & f
            convidx_all_incl = np.where(combined_filter_idx)[0]
        else:
            convidx_all_incl = np.arange(len(participant_baby))

        return {
            "participant_baby": participant_baby,
            "participant_included": participant_included,
            "pilotnames_all": pilotnames_all,
            "pilotnames_incl": pilotnames_incl,
            "participants_incl": pilotnames_incl,
            "convidx_all_incl": convidx_all_incl,
        }

    # -------------------------------------------------------------------------
    def get_eeg_data(self, segments_config=None, sample_mask=None, verbose=None):
        """Load preprocessed EEG epochs for each segment.

        Parameters
        ----------
        segments_config : dict, optional
            Per-segment loading configuration. Each key is a segment name with
            a dict containing: load, path, filename_suffix, process_labels,
            reshape_data, resample_freq, filter_params.
        sample_mask : list of str, optional
            Participant names to load (e.g. participant_info["participants_incl"]).
            If None, loads all participants (requires get_participant_info called first).
        verbose : bool, optional
            Whether to print verbose output during loading.

        Returns
        -------
        dict keyed by segment name. Each value has: epochs, data, trial_labels,
            times, ch_names, info, preprocessing. Arrays are indexed 0..n-1
            where n = len(sample_mask).
        """
        pilotnames = sample_mask if sample_mask is not None else self._pilotnames_all
        if pilotnames is None:
            raise RuntimeError("Call get_participant_info() first or provide sample_mask.")

        default_filter_params = {
            "l_freq": None,
            "h_freq": None,
            "bandpass": False,
            "filter_kwargs": {"method": "fir"},
            "decimate": 1,
        }

        if segments_config is None:
            segments_config = {
                "localizer": {
                    "load": True,
                    "path": os.path.join("Segments", "Localizer"),
                    "filename_suffix": "_Epochs.fif",
                    "process_labels": True,
                    "reshape_data": False,
                    "resample_freq": 100,
                },
                "resting": {
                    "load": True,
                    "path": os.path.join("Segments", "Resting"),
                    "filename_suffix": "_Epochs.fif",
                    "process_labels": False,
                    "reshape_data": True,
                    "resample_freq": 100,
                },
                "cued_replay": {
                    "load": True,
                    "path": os.path.join("Segments", "CuedReplay"),
                    "filename_suffix": "_Epochs.fif",
                    "process_labels": False,
                    "reshape_data": False,
                    "resample_freq": 100,
                },
                "pre_resting": {
                    "load": True,
                    "process_labels": False,
                    "reshape_data": True,
                    "resample_freq": 100,
                    "merge": [
                        {
                            "path": os.path.join("Segments", "PreResting"),
                            "filename_suffix": "_Epochs.fif",
                        },
                        {
                            "path": os.path.join("Segments", "PreResting"),
                            "filename_suffix": "_Break_Epochs.fif",
                        },
                    ],
                },
            }

        for segment in segments_config.values():
            if segment.get("resample_freq") is None:
                segment["filter_params"] = segment.get("filter_params", default_filter_params)

        eeg_data = {}

        for segment_name, config in segments_config.items():
            if not config["load"]:
                continue

            n = len(pilotnames)
            epochs_array = np.empty((n,), dtype=object)
            data_array = np.empty((n,), dtype=object)
            labels_array = np.empty((n,), dtype=object)

            for p, name in enumerate(pilotnames):
                # ── load: single file or merged from multiple sources ──────
                if "merge" in config:
                    parts = []
                    for src in config["merge"]:
                        fp = os.path.join(
                            self.preproc_dir,
                            src["path"],
                            f"{name}{src['filename_suffix']}",
                        )
                        if os.path.exists(fp):
                            parts.append(mne.read_epochs(fp, preload=True, verbose=verbose))
                    if not parts:
                        continue
                    for _ep in parts:
                        _ep.baseline = None
                        _ep.metadata = None
                    if len(parts) > 1:
                        tmin = max(ep.tmin for ep in parts)
                        tmax = min(ep.tmax for ep in parts)
                        parts = [ep.crop(tmin=tmin, tmax=tmax) for ep in parts]
                        epochs = mne.concatenate_epochs(parts)
                    else:
                        epochs = parts[0]
                else:
                    file_path = os.path.join(
                        self.preproc_dir,
                        config["path"],
                        f"{name}{config['filename_suffix']}",
                    )
                    if not os.path.exists(file_path):
                        continue
                    epochs = mne.read_epochs(file_path, preload=True, verbose=verbose)

                if config["resample_freq"] is None and "filter_params" in config:
                    filter_params = config["filter_params"]
                    if filter_params.get("bandpass", False):
                        epochs.filter(
                            l_freq=filter_params["l_freq"],
                            h_freq=filter_params["h_freq"],
                            **filter_params.get("filter_kwargs", {}),
                        )
                    else:
                        if filter_params.get("l_freq") is not None:
                            epochs.filter(
                                l_freq=filter_params["l_freq"],
                                h_freq=None,
                                **filter_params.get("filter_kwargs", {}),
                            )
                        if filter_params.get("h_freq") is not None:
                            epochs.filter(
                                l_freq=None,
                                h_freq=filter_params["h_freq"],
                                **filter_params.get("filter_kwargs", {}),
                            )
                    if filter_params.get("decimate", 1) > 1:
                        epochs.decimate(filter_params["decimate"])
                    elif filter_params.get("resample", 1) > 1:
                        epochs.resample(filter_params["resample"])
                elif config["resample_freq"] is not None:
                    epochs.resample(config["resample_freq"])

                epochs_array[p] = epochs

                if config["reshape_data"]:
                    data_array[p] = np.reshape(
                        np.transpose(epochs.copy().get_data(copy=False), (1, 0, 2)),
                        (len(epochs.ch_names), -1),
                    )
                    labels_array[p] = np.ones(len(epochs))
                else:
                    data_array[p] = epochs.copy().get_data(copy=False)
                    if config["process_labels"] and segment_name == "localizer":
                        trl_index_loc = np.vstack(
                            [
                                [1 if "apple" in a else 0 for a in epochs.metadata["Loc_Image"]],
                                [2 if "chair" in a else 0 for a in epochs.metadata["Loc_Image"]],
                                [3 if "face" in a else 0 for a in epochs.metadata["Loc_Image"]],
                            ]
                        )
                        labels_array[p] = trl_index_loc.sum(0)
                    else:
                        labels_array[p] = np.ones(len(epochs))

            ref_epoch = next(ep for ep in epochs_array if ep is not None)
            eeg_data[segment_name] = {
                "epochs": epochs_array,
                "data": data_array,
                "trial_labels": labels_array,
                "times": ref_epoch.times,
                "ch_names": ref_epoch.ch_names,
                "info": ref_epoch.info,
                "preprocessing": {
                    "resample_freq": config["resample_freq"],
                    "process_labels": config["process_labels"],
                    "reshape_data": config["reshape_data"],
                    "path": config.get("path", config.get("merge")),
                    "filename_suffix": config.get("filename_suffix", "merge"),
                },
            }

        return eeg_data

    # -------------------------------------------------------------------------
    def get_behavioral_data(self, eeg_data, participant_info, raw_dir):
        """Load behavioral data and compute per-segment statistics.

        Parameters
        ----------
        eeg_data : dict
            Output of get_eeg_data().
        participant_info : dict
            Output of get_participant_info().
        raw_dir : str or Path
            Root directory of raw data (must contain a 'ratings' sub-directory).

        Returns
        -------
        behavioral_data : dict
            Per-segment n_trials, s_length, mean/std statistics.
            Top-level keys also include info_names, mean_age, and
            behavioral_data["localizer"]["trial_ratings"] /
            behavioral_data["localizer"]["cohkappa_scores"].
        """
        from sklearn.metrics import cohen_kappa_score

        raw_dir = Path(raw_dir)
        segments = list(eeg_data.keys())
        pilotnames = participant_info["pilotnames_incl"]
        n_subs = len(pilotnames)

        behavioral_data = {name: {} for name in segments}

        # ── ratings + Cohen's kappa ──────────────────────────────────────────
        trial_ratings = pd.DataFrame()
        for name in tqdm(pilotnames, desc="Subject ratings loaded"):
            rater1 = pd.read_csv(
                raw_dir / "ratings" / "rater1" / f"Template_Localizer_{name}.csv",
                sep=";",
            ).dropna(subset="Block")
            rater1.insert(0, "Participant", name)

            rater2 = pd.read_excel(
                raw_dir / "ratings" / "rater2" / f"Template_Localizer_{name}.xlsx",
                engine="calamine",
                decimal=",",
            ).dropna(subset="Block")
            rater2.insert(0, "Participant", name)

            trial_ratings = pd.concat(
                [
                    trial_ratings,
                    pd.merge(
                        rater1,
                        rater2,
                        on=["Participant", "Block", "Trials"],
                        suffixes=("_rater1", "_rater2"),
                    ),
                ]
            )

        behavioral_data["localizer"]["trial_ratings"] = trial_ratings

        cohkappa_scores = []
        for name in pilotnames:
            mask = trial_ratings["Participant"] == name
            cohkappa_scores.append(
                cohen_kappa_score(
                    trial_ratings.loc[mask, "Attends_Bool_rater1"],
                    trial_ratings.loc[mask, "Attends_Bool_rater2"],
                )
            )
        behavioral_data["localizer"]["cohkappa_scores"] = cohkappa_scores

        # ── per-segment trial counts and recording lengths ───────────────────
        for seg in segments:
            seg_labels = np.unique(eeg_data[seg]["trial_labels"][0])
            behavioral_data[seg]["n_trials"] = np.zeros((n_subs, seg_labels.shape[0]))
            behavioral_data[seg]["s_length"] = np.zeros((n_subs, seg_labels.shape[0]))
            for p in tqdm(range(n_subs), desc=f"{seg} – subjects", leave=False):
                _, behavioral_data[seg]["n_trials"][p, :] = np.unique(
                    eeg_data[seg]["trial_labels"][p], return_counts=True
                )
                behavioral_data[seg]["s_length"][p, :] = (
                    np.prod(eeg_data[seg]["data"][p].shape[0:3:2]) / 100
                )

            if seg == "localizer":
                temp_avg = 1 - behavioral_data[seg]["n_trials"].sum(1) / 540
            elif seg == "cued_replay":
                temp_avg = 1 - behavioral_data[seg]["n_trials"].sum(1) / 50
            elif seg == "seq_learn":
                temp_avg = 1 - behavioral_data[seg]["n_trials"].sum(1) / 100
            else:
                temp_avg = behavioral_data[seg]["n_trials"].sum(1)

            behavioral_data[seg]["mean_trlrej"] = np.mean(temp_avg)
            behavioral_data[seg]["std_trlrej"] = np.std(temp_avg)
            behavioral_data[seg]["mean_length"] = np.mean(behavioral_data[seg]["s_length"])
            behavioral_data[seg]["std_length"] = np.std(behavioral_data[seg]["s_length"])

        # ── demographics ─────────────────────────────────────────────────────
        behavioral_data["info_names"] = (
            participant_info["participant_included"]["Age_Months"].astype(int).astype(str)
            + participant_info["participant_included"]["Gender"].astype(str)
        ).to_numpy()

        behavioral_data["mean_age"] = participant_info["participant_included"]["Age_Months"].mean()

        return behavioral_data
