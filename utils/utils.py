# -*- coding: utf-8 -*-
"""
Utility functions

Created on Mon Jan 29 16:15:01 2024
@author: Christopher Postzich
@github: mcpost
"""

import os
import numpy as np
from tqdm import trange, tqdm
import mne

import scipy.stats as scistats
from scipy.ndimage import label
from skimage import measure
from sklearn.linear_model import LinearRegression


# Channel Grid
chan_grid = {
    "F9": 0,
    "F7": 1,
    "F3": 2,
    "Fz": 3,
    "F4": 4,
    "F8": 5,
    "F10": 6,
    "FC5": 8,
    "FC3": 9,
    "FCz": 10,
    "FC4": 11,
    "FC6": 12,
    "T7": 14,
    "C3": 16,
    "C4": 18,
    "T8": 20,
    "CP5": 22,
    "CP3": 23,
    "CP4": 25,
    "CP6": 26,
    "TP9": 28,
    "P7": 29,
    "P3": 30,
    "Pz": 31,
    "P4": 32,
    "P8": 33,
    "TP10": 34,
    "POz": 38,
    "O1": 44,
    "Oz": 45,
    "O2": 46,
}


# Calculate Standard Error
def std_error(data, axis=0, ddof=1):
    return np.std(data, axis=axis, ddof=ddof) / np.sqrt(data.shape[axis])


def sign_flip_permtest(data, n_permutations=1000, clust_thresh_pval=0.05, **kwargs):
    """
    Perform cluster-based sign-flip permutation test for statistical inference.

    This function implements a non-parametric cluster-based permutation test using
    sign-flipping to assess statistical significance of effects in 1D (time series)
    or 2D (time-time matrices) data. It computes t-statistics against a chance level
    and identifies significant clusters using permutation-derived null distributions.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array with shape:
        - 2D: (n_subjects, n_timepoints) for time series analysis
        - 3D: (n_subjects, n_train_times, n_test_times) for temporal generalization
    n_permutations : int, default=1000
        Number of sign-flip permutations to generate null distribution.
    clust_thresh_pval : float, default=0.05
        P-value threshold for cluster formation. Samples exceeding this threshold
        (in t-value space) are considered for cluster membership.

    Keyword Arguments
    -----------------
    sided : str, default='two.sided'
        Type of test: 'greater' (positive effects), 'lesser' (negative effects),
        or 'two.sided' (both directions).
    chance_lev : float, default=0.0
        Chance level to test against. Data is shifted by this value before testing.
    test : str, default='cluster'
        Which extreme values to test against. "cluster" or "max".
    add_info : bool, default=False
        If True, returns additional information including surrogate cluster info
        and t-maps for both real and permuted data.
    time_info : list of dict, optional
        Time window specifications for cropping data before analysis.
        For 2D data: [dict(times=array, window=[start, end])]
        For 3D data: [dict(times=array, window=[start, end]),  # x-axis
                      dict(times=array, window=[start, end])]  # y-axis

    Returns
    -------
    real_clust_info : dict
        Dictionary with keys 'pos' and 'neg', each containing a list of cluster
        dictionaries with the following fields:
        - cluster_area: int, number of samples in the cluster
        - cluster_area_pval: float, p-value based on cluster area
        - cluster_tsum: float, sum of t-values within the cluster
        - cluster_tsum_pval: float, p-value based on cluster t-sum
        - cluster_index: array, indices of cluster samples (with time offset)

    If add_info=True, also returns:
        surrogate_clust_info : dict
            Null distribution cluster statistics for 'pos' and 'neg' directions.
        real_tmap : numpy.ndarray
            T-statistic map from the actual data.
        surrogate_tmap : numpy.ndarray
            T-statistic maps from all permutations.

    Notes
    -----
    The sign-flip permutation approach randomly flips the sign of each subject's
    data to create a null distribution under the assumption that the true effect
    is zero. This preserves the temporal autocorrelation structure within subjects.

    Examples
    --------
    >>> # 1D time series test
    >>> data = np.random.randn(20, 100) + 0.3  # 20 subjects, 100 timepoints
    >>> clusters = sign_flip_permtest(data, n_permutations=1000, chance_lev=0.0)
    >>>
    >>> # 2D temporal generalization test with time window
    >>> data_2d = np.random.randn(20, 80, 120)  # 20 subjects, 80x120 matrix
    >>> time_info = [dict(times=train_times, window=[0.0, 0.5]),
    ...              dict(times=test_times, window=[-0.1, 1.0])]
    >>> clusters = sign_flip_permtest(data_2d, time_info=time_info, chance_lev=1/3)
    """
    # Extract optional parameters with defaults
    sided = kwargs.get("sided", "two.sided")
    chance_lev = kwargs.get("chance_lev", 0.0)
    add_info = kwargs.get("add_info", False)
    time_info = kwargs.get("time_info", None)
    test = kwargs.get("test", "cluster")

    # Initialize index offsets (used for cluster coordinate reporting)
    timex_idx = [0, 0]
    timey_idx = [0, 0]

    # Crop time window if specified
    if time_info:
        if not isinstance(time_info, list):
            time_info = [time_info]

        if len(time_info) == 2 and data.ndim == 3:
            # 2D case: temporal generalization matrix
            timex = time_info[0]["times"]
            windx = time_info[0]["window"]
            timex_idx = [
                np.argmin(np.abs(timex - windx[0])),
                np.argmin(np.abs(timex - windx[1])) + 1,
            ]
            timey = time_info[1]["times"]
            windy = time_info[1]["window"]
            timey_idx = [
                np.argmin(np.abs(timey - windy[0])),
                np.argmin(np.abs(timey - windy[1])) + 1,
            ]
            data = data[:, timex_idx[0] : timex_idx[1], timey_idx[0] : timey_idx[1]]
        elif len(time_info) == 1 and data.ndim == 2:
            # 1D case: time series
            timex = time_info[0]["times"]
            windx = time_info[0]["window"]
            timex_idx = [
                np.argmin(np.abs(timex - windx[0])),
                np.argmin(np.abs(timex - windx[1])),
            ]
            data = data[:, timex_idx[0] : timex_idx[1]]

    # Degrees of freedom
    df = data.shape[0] - 1

    # Compute real t-map and initialize surrogate storage
    if data.ndim == 3:
        dims = [1, 1]
        surrogate_tmap = np.zeros((n_permutations, data.shape[-2], data.shape[-1]))
        real_tmap = scistats.ttest_1samp(data - chance_lev, 0)[0]
    else:
        dims = [1]
        surrogate_tmap = np.zeros((n_permutations, data.shape[-1]))
        real_tmap = scistats.ttest_1samp(data - chance_lev, 0)[0][np.newaxis, :]

    surrogate_mean_map = np.zeros_like(surrogate_tmap)

    # Generate surrogate t-maps via sign-flipping
    for perm in trange(n_permutations, desc="Sign Flip Permutations"):
        sign_flip = np.sign(2 * np.random.rand(data.shape[0], *dims) - 1)
        surr_data = sign_flip * (data - chance_lev)
        surrogate_tmap[perm, :], _ = scistats.ttest_1samp(surr_data, 0)
        surrogate_mean_map[perm] = surr_data.mean(0)

    if test == "cluster":
        # ------------------------------------------------------------------ #
        # Cluster-based null distribution                                     #
        # ------------------------------------------------------------------ #
        surrogate_clust_info = {
            "pos": np.zeros((n_permutations, 2)),
            "neg": np.zeros((n_permutations, 2)),
        }

        t_thresh_pos = scistats.t.ppf(1 - clust_thresh_pval, df)
        t_thresh_neg = scistats.t.ppf(clust_thresh_pval, df)

        for perm in trange(n_permutations, desc="Find Surrogate Clusters"):
            temp = surrogate_tmap[perm] if data.ndim == 3 else surrogate_tmap[perm][np.newaxis, :]

            if sided in ("greater", "two.sided"):
                clust_map_pos = temp >= t_thresh_pos
                labels = measure.label(clust_map_pos)
                props = measure.regionprops(labels)
                if labels.sum() > 0:
                    surrogate_clust_info["pos"][perm, 0] = np.max([cl.area for cl in props])
                    surrogate_clust_info["pos"][perm, 1] = np.max(
                        [temp[cl.coords[:, 0], cl.coords[:, 1]].sum() for cl in props]
                    )

            if sided in ("lesser", "two.sided"):
                clust_map_neg = temp <= t_thresh_neg
                labels = measure.label(clust_map_neg)
                props = measure.regionprops(labels)
                if labels.sum() > 0:
                    surrogate_clust_info["neg"][perm, 0] = np.max([cl.area for cl in props])
                    surrogate_clust_info["neg"][perm, 1] = np.max(
                        [temp[cl.coords[:, 0], cl.coords[:, 1]].sum() for cl in props]
                    )

        # Find clusters in real data and compute p-values
        stat_info = dict(pos=[], neg=[])

        clust_map_pos = real_tmap >= t_thresh_pos
        labels = measure.label(clust_map_pos)
        props = measure.regionprops(labels)
        pos_clusters = []
        for cl in props:
            cluster_tsum = np.sum(real_tmap[cl.coords[:, 0], cl.coords[:, 1]])
            pos_clusters.append(
                {
                    "cluster_area": int(cl.area),
                    "cluster_area_pval": np.mean(cl.area >= surrogate_clust_info["pos"][:, 0]),
                    "cluster_tsum": cluster_tsum,
                    "cluster_tsum_pval": np.mean(cluster_tsum >= surrogate_clust_info["pos"][:, 1]),
                    "cluster_stat": cluster_tsum,
                    "cluster_stat_pval": float(
                        np.mean(surrogate_clust_info["pos"][:, 1] >= cluster_tsum)
                    ),
                    "cluster_index": cl.coords + [timex_idx[0], timey_idx[0]],
                }
            )
        stat_info["pos"] = pos_clusters

        clust_map_neg = real_tmap <= t_thresh_neg
        labels = measure.label(clust_map_neg)
        props = measure.regionprops(labels)
        neg_clusters = []
        for cl in props:
            cluster_tsum = np.sum(real_tmap[cl.coords[:, 0], cl.coords[:, 1]])
            neg_clusters.append(
                {
                    "cluster_area": int(cl.area),
                    "cluster_area_pval": np.mean(cl.area >= surrogate_clust_info["neg"][:, 0]),
                    "cluster_tsum": cluster_tsum,
                    "cluster_tsum_pval": np.mean(cluster_tsum <= surrogate_clust_info["neg"][:, 1]),
                    "cluster_stat": cluster_tsum,
                    "cluster_stat_pval": float(
                        np.mean(surrogate_clust_info["neg"][:, 1] <= cluster_tsum)
                    ),
                    "cluster_index": cl.coords + [timex_idx[0], timey_idx[0]],
                }
            )
        stat_info["neg"] = neg_clusters

        if add_info:
            return stat_info, surrogate_clust_info, real_tmap, surrogate_tmap
        return stat_info

    elif test == "max":
        # ------------------------------------------------------------------ #
        # Max-statistic null distribution                                     #
        # ------------------------------------------------------------------ #
        tmap_use = real_tmap if data.ndim == 3 else real_tmap[0]

        ax_max = tuple(range(1, surrogate_tmap.ndim))

        if sided == "greater":
            null_max = surrogate_tmap.max(axis=ax_max)
            real_max = float(tmap_use.max())
            threshold = float(np.percentile(null_max, (1 - clust_thresh_pval) * 100))
            sig_arr = tmap_use >= threshold
            null_max_mean = surrogate_mean_map.max(axis=ax_max)
            threshold_data = float(np.percentile(null_max_mean, (1 - clust_thresh_pval) * 100))
        elif sided == "lesser":
            null_max = (-surrogate_tmap).max(axis=ax_max)
            real_max = float((-tmap_use).max())
            threshold = float(np.percentile(null_max, (1 - clust_thresh_pval) * 100))
            sig_arr = tmap_use <= -threshold
            null_max_mean = (-surrogate_mean_map).max(axis=ax_max)
            threshold_data = float(-np.percentile(null_max_mean, (1 - clust_thresh_pval) * 100))
        else:  # two.sided: use max(|t|)
            null_max = np.abs(surrogate_tmap).max(axis=ax_max)
            real_max = float(np.abs(tmap_use).max())
            threshold = float(np.percentile(null_max, (1 - clust_thresh_pval / 2) * 100))
            sig_arr = np.abs(tmap_use) >= threshold
            null_max_mean = np.abs(surrogate_mean_map).max(axis=ax_max)
            threshold_data = float(np.percentile(null_max_mean, (1 - clust_thresh_pval / 2) * 100))

        if data.ndim == 3:
            sig_indices = np.argwhere(sig_arr) + np.array([timex_idx[0], timey_idx[0]])
        else:
            sig_indices = np.argwhere(sig_arr) + timex_idx[0]

        stat_info = {
            "max_dist": null_max,
            "max_real": real_max,
            "pval": float(np.mean(null_max >= real_max)),
            "threshold": threshold,  # FWE critical t-value
            "threshold_data": threshold_data,  # FWE critical value in data units
            "sig_times": sig_indices,
        }

        if add_info:
            return stat_info, null_max, real_tmap, surrogate_tmap
        return stat_info

    else:
        raise ValueError(f"test must be 'cluster' or 'max', got {test!r}")


def label_switch_permtest(data_a, data_b, n_permutations=1000, clust_thresh_pval=0.05, **kwargs):
    """
    Between-group permutation test via label switching (two-sample t-statistic).

    The null distribution is built by randomly reassigning observations to the
    two groups and recomputing the two-sample t-statistic at every lag. This is
    the correct non-parametric test when sign-flipping is not applicable (i.e.
    for independent-groups comparisons).

    Parameters
    ----------
    data_a : ndarray, shape (n_a, n_lags)
        Net sequenceness for group A (already filtered to valid subjects).
    data_b : ndarray, shape (n_b, n_lags)
        Net sequenceness for group B.
    n_permutations : int, default 1000
        Number of label-switch permutations.
    clust_thresh_pval : float, default 0.05
        Cluster-formation or max-stat threshold p-value.

    Keyword Arguments
    -----------------
    sided : str, default 'two.sided'
        'greater', 'lesser', or 'two.sided'.
    test : str, default 'cluster'
        'cluster' (cluster-based FWE) or 'max' (max-statistic FWE).
    add_info : bool, default False
        If True, return the 4-tuple (stat_info, null_max, real_tmap, surrogate_tmap).

    Returns
    -------
    stat_info : dict
        Same structure as sign_flip_permtest — keys depend on `test`.
    If add_info=True, also:
        null_max        : (n_permutations,) null distribution of the max statistic
        real_tmap       : (1, n_lags) two-sample t-map for the real data
        surrogate_tmap  : (n_permutations, n_lags) permuted t-maps
    """
    sided = kwargs.get("sided", "two.sided")
    add_info = kwargs.get("add_info", False)
    test = kwargs.get("test", "cluster")

    n_a = data_a.shape[0]
    n_b = data_b.shape[0]
    df = n_a + n_b - 2
    n_lag = data_a.shape[1]

    # Real two-sample t-map: shape (1, n_lags) to mirror sign_flip_permtest convention
    real_t, _ = scistats.ttest_ind(data_a, data_b, axis=0)  # (n_lags,)
    real_tmap = real_t[np.newaxis, :]  # (1, n_lags)

    combined = np.vstack([data_a, data_b])  # (n_a+n_b, n_lags)
    surrogate_tmap = np.zeros((n_permutations, n_lag))
    surrogate_mean_map = np.zeros((n_permutations, n_lag))

    for perm in trange(n_permutations, desc="Label Switch Permutations"):
        idx = np.random.permutation(n_a + n_b)
        surr_t, _ = scistats.ttest_ind(combined[idx[:n_a]], combined[idx[n_a:]], axis=0)
        surrogate_tmap[perm] = surr_t
        surrogate_mean_map[perm] = combined[idx[:n_a]].mean(0) - combined[idx[n_a:]].mean(0)

    tmap_use = real_tmap[0]  # (n_lags,) — used in both branches below

    if test == "cluster":
        # ------------------------------------------------------------------
        # Cluster-based null distribution
        # ------------------------------------------------------------------
        surrogate_clust_info = {
            "pos": np.zeros((n_permutations, 2)),
            "neg": np.zeros((n_permutations, 2)),
        }

        t_thresh_pos = scistats.t.ppf(1 - clust_thresh_pval, df)
        t_thresh_neg = scistats.t.ppf(clust_thresh_pval, df)

        for perm in trange(n_permutations, desc="Find Surrogate Clusters"):
            temp = surrogate_tmap[perm][np.newaxis, :]  # (1, n_lags) for measure.label

            if sided in ("greater", "two.sided"):
                labels = measure.label(temp >= t_thresh_pos)
                props = measure.regionprops(labels)
                if np.sum(labels) > 0:
                    surrogate_clust_info["pos"][perm, 0] = max(cl.area for cl in props)
                    surrogate_clust_info["pos"][perm, 1] = max(
                        temp[cl.coords[:, 0], cl.coords[:, 1]].sum() for cl in props
                    )

            if sided in ("lesser", "two.sided"):
                labels = measure.label(temp <= t_thresh_neg)
                props = measure.regionprops(labels)
                if np.sum(labels) > 0:
                    surrogate_clust_info["neg"][perm, 0] = max(cl.area for cl in props)
                    surrogate_clust_info["neg"][perm, 1] = max(
                        temp[cl.coords[:, 0], cl.coords[:, 1]].sum() for cl in props
                    )

        stat_info = {"pos": [], "neg": []}
        real_tmap_2d = real_tmap  # (1, n_lags)

        for direction, thresh, surr_key in (
            ("pos", t_thresh_pos, "pos"),
            ("neg", t_thresh_neg, "neg"),
        ):
            mask = real_tmap_2d >= thresh if direction == "pos" else real_tmap_2d <= thresh
            labels = measure.label(mask)
            props = measure.regionprops(labels)
            clusters = []
            for cl in props:
                cluster_tsum = float(real_tmap_2d[cl.coords[:, 0], cl.coords[:, 1]].sum())
                if direction == "pos":
                    stat_pval = float(np.mean(surrogate_clust_info[surr_key][:, 1] >= cluster_tsum))
                else:
                    stat_pval = float(np.mean(surrogate_clust_info[surr_key][:, 1] <= cluster_tsum))
                clusters.append(
                    {
                        "cluster_area": int(cl.area),
                        "cluster_area_pval": float(
                            np.mean(cl.area >= surrogate_clust_info[surr_key][:, 0])
                        ),
                        "cluster_tsum": cluster_tsum,
                        "cluster_tsum_pval": stat_pval,
                        "cluster_stat": cluster_tsum,
                        "cluster_stat_pval": stat_pval,
                        "cluster_index": cl.coords,
                    }
                )
            stat_info[direction] = clusters

        if add_info:
            return stat_info, surrogate_clust_info, real_tmap, surrogate_tmap
        return stat_info

    elif test == "max":
        # ------------------------------------------------------------------
        # Max-statistic null distribution
        # ------------------------------------------------------------------
        ax_max = 1  # axis over lags

        if sided == "greater":
            null_max = surrogate_tmap.max(axis=ax_max)
            real_max = float(tmap_use.max())
            threshold = float(np.percentile(null_max, (1 - clust_thresh_pval) * 100))
            sig_arr = tmap_use >= threshold
            null_max_mean = surrogate_mean_map.max(axis=ax_max)
            threshold_data = float(np.percentile(null_max_mean, (1 - clust_thresh_pval) * 100))
        elif sided == "lesser":
            null_max = (-surrogate_tmap).max(axis=ax_max)
            real_max = float((-tmap_use).max())
            threshold = float(np.percentile(null_max, (1 - clust_thresh_pval) * 100))
            sig_arr = tmap_use <= -threshold
            null_max_mean = (-surrogate_mean_map).max(axis=ax_max)
            threshold_data = float(-np.percentile(null_max_mean, (1 - clust_thresh_pval) * 100))
        else:  # two.sided
            null_max = np.abs(surrogate_tmap).max(axis=ax_max)
            real_max = float(np.abs(tmap_use).max())
            threshold = float(np.percentile(null_max, (1 - clust_thresh_pval) * 100))
            sig_arr = np.abs(tmap_use) >= threshold
            null_max_mean = np.abs(surrogate_mean_map).max(axis=ax_max)
            threshold_data = float(np.percentile(null_max_mean, (1 - clust_thresh_pval) * 100))

        sig_indices = np.argwhere(sig_arr)

        stat_info = {
            "max_dist": null_max,
            "max_real": real_max,
            "pval": float(np.mean(null_max >= real_max)),
            "threshold": threshold,
            "threshold_data": threshold_data,
            "sig_times": sig_indices,
        }

        if add_info:
            return stat_info, null_max, real_tmap, surrogate_tmap
        return stat_info

    else:
        raise ValueError(f"test must be 'cluster' or 'max', got {test!r}")
