import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from mne.baseline import rescale
from mne.decoding import LinearModel, SlidingEstimator, cross_val_multiscore, get_coef

from utils.utils import sign_flip_permtest


def run_decode(eeg_data, sample_mask, params):
    """
    Train a time-resolved sliding classifier on localizer data and run a
    group-level sign-flip permutation test.

    Parameters
    ----------
    eeg_data : dict
        Output of Things_Importer.get_eeg_data(); must contain the "localizer" key.
    sample_mask : array-like
        Integer indices of participants (0-based, pre-pruned).
    params : dict
        Cs, solver, penalty, max_iter, CV, N_SIGN_PERMS

    Returns
    -------
    classifier_data : dict
        clf, scoring, spatial_patterns, spatial_filters, performance, group_stats
    """
    n_subs = len(sample_mask)
    Cs = params["Cs"]
    solver = params["solver"]
    penalty = params["penalty"]
    max_iter = params["max_iter"]
    CV = params["CV"]
    N_SIGN_PERMS = params["N_SIGN_PERMS"]
    CLUST_THRESH_PVAL = params["CLUST_THRESH_PVAL"]

    classifier_data = {
        "clf": [[]] * n_subs,
        "scoring": "roc_auc_ovr",
        "spatial_patterns": [[]] * n_subs,
        "spatial_filters": [[]] * n_subs,
        "performance": np.ndarray((n_subs, CV, len(eeg_data["localizer"]["times"]))),
        "group_stats": {
            "cluster_dict": {},
            "n_perms": N_SIGN_PERMS,
        },
    }

    for p, data, label in tqdm(
        zip(
            range(n_subs),
            eeg_data["localizer"]["data"],
            eeg_data["localizer"]["trial_labels"],
            strict=True,
        ),
        total=n_subs,
        desc="Participants cross-validated",
    ):
        if data is not None:
            data_bl = rescale(
                data,
                eeg_data["localizer"]["times"],
                (-0.2, 0.0),
                "mean",
                verbose=False,
            )

            clf = make_pipeline(
                StandardScaler(),
                LinearModel(
                    LogisticRegression(
                        C=Cs[p],
                        solver=solver,
                        penalty=penalty,
                        max_iter=max_iter,
                    )
                ),
            )
            classifier_data["clf"][p] = clf

            time_decod = SlidingEstimator(
                OneVsRestClassifier(clf),
                n_jobs=None,
                scoring=classifier_data["scoring"],
                verbose=False,
            )
            time_decod.fit(data_bl, label)
            classifier_data["spatial_patterns"][p] = get_coef(
                time_decod, "patterns_", inverse_transform=True
            )
            classifier_data["spatial_filters"][p] = get_coef(
                time_decod, "filters_", inverse_transform=False
            )
            classifier_data["performance"][p, :, :] = cross_val_multiscore(
                time_decod, data_bl, label, cv=CV, verbose=False
            )

    classifier_data["group_stats"]["cluster_dict"] = sign_flip_permtest(
        classifier_data["performance"],
        N_SIGN_PERMS,
        CLUST_THRESH_PVAL,
        chance_lev=0.5,
    )

    return classifier_data
