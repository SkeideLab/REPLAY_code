import numpy as np
from tqdm import tqdm

from utils.CrossCorrelation import CrossCorrelationSequenceness
from utils.utils import sign_flip_permtest, label_switch_permtest


def run_xcorr(decode_data, params, segments):
    """
    Cross-correlation sequenceness analysis with embedded sign-flip permutation tests.

    Parameters
    ----------
    decode_data : dict
        Output of run_xdecode(); must contain decode_data[seg]["probabilities"]
        and decode_data["classes"].
    params : dict
        MAX_LAG, tm, N_SIGN_PERMS
    segments : list of str

    Returns
    -------
    xcorr_data : dict
        xcorr_data[seg] contains raw sequenceness arrays plus perm-test results
        embedded directly (skip, seq_net, perm_net, stat_info, tmap, threshold,
        threshold_data).
    """
    MAX_LAG = params["MAX_LAG"]
    tm = params["tm"]
    N_SIGN_PERMS = params["N_SIGN_PERMS"]
    classes = decode_data["classes"]

    xcorr_model = CrossCorrelationSequenceness(max_lag=MAX_LAG)
    tm_perms = xcorr_model._get_tm_permutations(tm)
    n_perms = len(tm_perms)

    xcorr_data = {}

    for name in tqdm(segments, desc="XCorr sequenceness"):
        probs = decode_data[name]["probabilities"]
        n_subs = len(probs)
        xcorr_data[name] = {
            "sequenceness": np.nan * np.ones((n_subs, 3, MAX_LAG)),
            "permutations": np.nan * np.ones((n_subs, n_perms, 3, MAX_LAG)),
            "empirical_tm": np.nan * np.ones((n_subs, 3, 3, MAX_LAG)),
        }

        for p, prob in enumerate(tqdm(probs, desc=f"  {name} – participants", leave=False)):
            if not np.any(prob):
                continue

            temp1 = np.zeros((prob.shape[0], 3, MAX_LAG))
            temp2 = np.zeros((prob.shape[0], n_perms, 3, MAX_LAG))
            temp3 = np.zeros((prob.shape[0], 3, 3, MAX_LAG))
            for pr, cur_prob in enumerate(prob[:, :, :]):
                temp1[pr, :, :] = xcorr_model.fit(cur_prob, classes, model_tm=tm)
                temp2[pr, :, :] = xcorr_model.permutations(cur_prob, classes, model_tm=tm)
                temp3[pr, :, :, :] = xcorr_model.xcorr_
            xcorr_data[name]["sequenceness"][p] = temp1.mean(0)
            xcorr_data[name]["permutations"][p] = temp2.mean(0)
            xcorr_data[name]["empirical_tm"][p] = temp3.mean(0)

    for name in tqdm(segments, desc="Permutation tests"):
        skip = ~np.isnan(xcorr_data[name]["sequenceness"][:, 0, 0])
        seq_net = xcorr_data[name]["sequenceness"][skip, -1, :]
        perm_net = xcorr_data[name]["permutations"][skip, :, -1, :]

        stat_info, _, tmap, _ = sign_flip_permtest(
            seq_net,
            N_SIGN_PERMS,
            0.025,
            chance_lev=0.0,
            test="max",
            add_info=True,
        )

        xcorr_data[name]["skip"] = skip
        xcorr_data[name]["seq_net"] = seq_net
        xcorr_data[name]["perm_net"] = perm_net
        xcorr_data[name]["stat_info"] = stat_info
        xcorr_data[name]["tmap"] = tmap.squeeze()
        xcorr_data[name]["threshold"] = stat_info["threshold"]
        xcorr_data[name]["threshold_data"] = stat_info["threshold_data"]

    return xcorr_data


def run_group_comparison(xcorr_data_old, xcorr_data_young, params, segments):
    """
    Two-sample label-switch permutation test comparing old vs young groups.

    Returns
    -------
    xcorr_data_diff : dict
        xcorr_data_diff[seg] = {seq_diff, seq_old, seq_young, stat_info,
                                 tmap, surrogate_tmap, threshold, threshold_data}
    """
    N_SIGN_PERMS = params["N_SIGN_PERMS"]
    MAX_PVAL = params["MAX_PVAL"]
    xcorr_data_diff = {}

    for name in segments:
        seq_old = xcorr_data_old[name]["seq_net"]
        seq_young = xcorr_data_young[name]["seq_net"]

        stat_info, null_max, real_tmap, surrogate_tmap = label_switch_permtest(
            seq_old,
            seq_young,
            N_SIGN_PERMS,
            clust_thresh_pval=MAX_PVAL,
            test="max",
            sided="two.sided",
            add_info=True,
        )
        xcorr_data_diff[name] = {
            "seq_diff": seq_old.mean(0) - seq_young.mean(0),
            "seq_old": seq_old,
            "seq_young": seq_young,
            "stat_info": stat_info,
            "tmap": real_tmap.squeeze(),
            "surrogate_tmap": surrogate_tmap,
            "threshold": stat_info["threshold"],
            "threshold_data": stat_info["threshold_data"],
        }

    return xcorr_data_diff


def run_contrasts(xcorr_data, params, base_seg, comparison_segs):
    """
    Within-group one-sided label-switch permutation test: base_seg > each
    comparison segment.

    Returns
    -------
    xcorr_contr : dict
        xcorr_contr[comp_seg] = {seq_diff, seq_base, seq_comp, stat_info,
                                  tmap, surrogate_tmap, threshold, threshold_data}
    """
    N_SIGN_PERMS = params["N_SIGN_PERMS"]
    MAX_PVAL = params["MAX_PVAL"]
    xcorr_contr = {}

    base_net = xcorr_data[base_seg]["seq_net"]

    for comp_seg in comparison_segs:
        comp_net = xcorr_data[comp_seg]["seq_net"]

        stat_info, null_max, real_tmap, surrogate_tmap = label_switch_permtest(
            base_net,
            comp_net,
            N_SIGN_PERMS,
            clust_thresh_pval=MAX_PVAL,
            test="max",
            sided="greater",
            add_info=True,
        )
        xcorr_contr[comp_seg] = {
            "seq_diff": base_net.mean(0) - comp_net.mean(0),
            "seq_base": base_net,
            "seq_comp": comp_net,
            "stat_info": stat_info,
            "tmap": real_tmap.squeeze(),
            "surrogate_tmap": surrogate_tmap,
            "threshold": stat_info["threshold"],
            "threshold_data": stat_info["threshold_data"],
        }

    return xcorr_contr
