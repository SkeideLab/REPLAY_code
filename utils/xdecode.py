import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.CrossDecoding_MEEG import CrossDecoding_MEEG


def run_xdecode(eeg_data, sample_mask, params, segments):
    """
    Cross-decode: train on localizer, predict on each segment.

    Parameters
    ----------
    eeg_data : dict
        Output of Things_Importer.get_eeg_data().
    sample_mask : array-like
        Integer indices of participants (0-based, pre-pruned).
    params : dict
        Cs, solver, penalty, max_iter, training_timepoint, n_classes
    segments : list of str

    Returns
    -------
    decode_data : dict
        decode_data[seg] = {decoder, probabilities, decision_values, betas}
        decode_data["classes"] = class labels from the first valid decoder
    """
    n_subs = len(sample_mask)
    Cs = params["Cs"]
    solver = params["solver"]
    penalty = params["penalty"]
    max_iter = params["max_iter"]
    training_timepoint = params["training_timepoint"]
    n_classes = params["n_classes"]

    decode_data = {}

    for name in tqdm(segments, desc="Segments"):
        decode_data[name] = {
            "decoder": [[]] * n_subs,
            "probabilities": [[]] * n_subs,
            "decision_values": [[]] * n_subs,
            "betas": [[]] * n_subs,
        }

        for p, x_train, y_train, x_test in zip(
            range(n_subs),
            eeg_data["localizer"]["data"],
            eeg_data["localizer"]["trial_labels"],
            eeg_data[name]["data"],
            strict=True,
        ):
            if x_test is not None:
                clf = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "classifier",
                            LogisticRegression(
                                C=Cs[p],
                                solver=solver,
                                penalty=penalty,
                                max_iter=max_iter,
                            ),
                        ),
                    ]
                )

                decode_data[name]["decoder"][p] = CrossDecoding_MEEG(
                    base_estimator=clf,
                    train_times=eeg_data["localizer"]["times"],
                    test_times=eeg_data[name]["times"],
                    training_timepoint=training_timepoint,
                    baseline_timepoint=-0.1,
                    multi_model=True,
                    include_zero=False,
                    random_state=42,
                    verbose=True,
                )

                decode_data[name]["decoder"][p].fit(x_train, y_train)

                decode_data[name]["probabilities"][p] = decode_data[name]["decoder"][
                    p
                ].predict_proba(x_test)

                decode_data[name]["decision_values"][p] = decode_data[name]["decoder"][
                    p
                ].decision_function(x_test)

                decode_data[name]["betas"][p] = [[]] * len(
                    decode_data[name]["decoder"][p].decoders_
                )
                for it, tp in enumerate(decode_data[name]["decoder"][p].decoders_.values()):
                    if decode_data[name]["decoder"][p].multi_model:
                        temp = np.zeros((n_classes, x_train.shape[1]))
                        for id, dec in enumerate(tp.values()):
                            temp[id, :] = dec["classifier"].coef_
                        decode_data[name]["betas"][p][it] = temp
                    else:
                        decode_data[name]["betas"][p][it] = tp["singlemodel"]["classifier"].coef_

    # store class labels from the first valid decoder found
    decode_data["classes"] = None
    for name in segments:
        for dec in decode_data[name]["decoder"]:
            if dec != []:
                decode_data["classes"] = dec.classes_
                break
        if decode_data["classes"] is not None:
            break

    return decode_data
