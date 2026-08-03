# Utils Module

Utility functions and classes for the REPLAY EEG analysis pipeline.

## Modules

### paths.py

Centralized path configuration. Resolved automatically:

- `BASE_DIR` — repo root (parent of `utils/`)
- `UTILS_DIR` — `BASE_DIR / "utils"`
- `INFO_DIR` — `BASE_DIR / "additional_data"`

**Edit these placeholders before running the pipeline:**
- `RAW_DIR` — raw data root (attention-rating files)
- `PREPROC_DIR` — preprocessed EEG segment root
- `FIGURES_DIR` — reserved, currently unused
- `GRAPHICS_DIR` — figure output root (create `Results/` and `Paradigm/` subfolders)

---

### imports.py

Data import functionality for the REPLAY-Things experiment.

**`Things_Importer(info_dir, preproc_dir, labels=["Apple", "Chair", "Face"])`**

```python
importer = Things_Importer(INFO_DIR, PREPROC_DIR)

participant_info = importer.get_participant_info(participant_filters=None)
eeg_data = importer.get_eeg_data(segments_config=None, sample_mask=None, verbose=None)
behavioral = importer.get_behavioral_data(eeg_data, participant_info)
```

- **`get_participant_info(participant_filters=None)`** — reads `participants_info.xlsx`, computes leap-year-corrected age in months from `Birthday`/`Date`, and applies inclusion filters (`baby_only`, `include_only`, `min_age_months`, `max_age_months`, `require_localizer`, `require_sequence`, `custom_filter`). Returns a dict with `participant_baby`, `participant_included`, `pilotnames_all`, `pilotnames_incl`, `participants_incl` (alias), `convidx_all_incl`.
- **`get_eeg_data(segments_config=None, sample_mask=None, verbose=None)`** — loads (and optionally merges/filters/decimates) MNE epochs per segment for each participant in `sample_mask`. Each returned segment dict has `epochs`, `data`, `trial_labels`, `times`, `ch_names`, `info`, `preprocessing`. Arrays are indexed `0..n-1` over `sample_mask`, not the full participant list.
- **`get_behavioral_data(eeg_data, participant_info)`** — summarizes localizer trial counts and resting duration per participant.

**`get_additional_data(participant_info, sample_mask, eeg_data, raw_dir, save_to_path="")`**

Loads and cross-checks rater1/rater2 localizer attention ratings, computes per-segment trial counts / rejection rates / resting duration, and (if `"seq_learn"` is among the loaded segments) writes a `trial_info_*.csv` summary to `save_to_path`.

---

### decode.py

**`run_decode(eeg_data, sample_mask, params)`**

Trains a time-resolved sliding classifier (`StandardScaler` + `LinearModel(LogisticRegression)` wrapped in `OneVsRestClassifier` + `SlidingEstimator`, scored via `roc_auc_ovr`) on baseline-corrected localizer data for each participant, then runs a group-level sign-flip permutation test against chance (0.5).

`params` requires: `Cs` (per-participant regularization list), `solver`, `penalty`, `max_iter`, `CV`, `N_SIGN_PERMS`, `CLUST_THRESH_PVAL`.

Returns `classifier_data` with `clf`, `scoring`, `spatial_patterns`, `spatial_filters`, `performance` (n_subs × CV × n_times), `group_stats["cluster_dict"]`.

---

### CrossDecoding_MEEG.py

Cross-temporal decoding classifier (scikit-learn compatible: `BaseEstimator`, `ClassifierMixin`).

```python
class CrossDecoding_MEEG:
    def __init__(
        self,
        base_estimator,
        train_times=None,
        test_times=None,
        training_timepoint=None,
        baseline_timepoint=None,
        multi_model=False,
        include_zero=True,
        n_jobs=1,
        random_state=None,
        verbose=True,
    ): ...

    def fit(self, X, y):
        """Train one classifier per training timepoint (or one, if training_timepoint is set)."""

    def predict_proba(self, X):
        """Predict probabilities across all test timepoints."""

    def predict(self, X): ...

    def decision_function(self, X): ...

    def transform(self, X):
        """Alias for predict_proba."""
```

- `training_timepoint`: single time (seconds) to train on; if `None`, trains one classifier per timepoint (full cross-temporal generalization).
- `baseline_timepoint`: a single time, or `[min, max, n_draws]` to sample baseline trials (labeled `0`) from a time range, augmenting training data.
- `multi_model`: train one binary (one-vs-rest) classifier per class instead of a single multiclass model.
- Output shape depends on whether one or multiple training timepoints were used (see docstrings in the module).

---

### xdecode.py

**`run_xdecode(eeg_data, sample_mask, params, segments)`**

For each participant and each segment, trains a `CrossDecoding_MEEG` on localizer data at `params["training_timepoint"]` (baseline-augmented at `-0.1` s, `multi_model=True`, `include_zero=False`) and predicts stimulus probabilities/decision values on that segment.

`params` requires: `Cs`, `solver`, `penalty`, `max_iter`, `training_timepoint`, `n_classes`.

Returns `decode_data[seg] = {decoder, probabilities, decision_values, betas}` plus `decode_data["classes"]`.

---

### CrossCorrelation.py

Cross-correlation based sequenceness analysis — the cross-correlation analogue of TDLM, using the same transition-matrix convention.

**`create_transition_matrix(label, transitions)`** → `(tm_forward, tm_backward)`

**`class CrossCorrelationSequenceness(max_lag=50)`**

```python
model = CrossCorrelationSequenceness(max_lag=50)
sequenceness = model.fit(X, classes, model_tm=tm)  # (3, max_lag): [forward, backward, net]
null_dist = model.permutations(X, classes)  # (n_perms, 3, max_lag)
```

- `fit(X, classes, model_tm=...)` — `X` is `(n_states, n_timepoints)` stimulus-probability series; computes the lagged Pearson cross-correlation tensor `xcorr_[i, j, lag]` and projects it onto the forward/backward pairs defined by `model_tm`.
- `permutations(X, classes, model_tm_perms=...)` — reuses the cached `xcorr_` tensor and projects it onto all valid row/column permutations of `model_tm` (auto-generated via `_get_tm_permutations` if not supplied) to build a null distribution.

---

### xcorr.py

**`run_xcorr(decode_data, params, segments)`**

Computes cross-correlation sequenceness (via `CrossCorrelationSequenceness`) per participant and segment, then runs a max-statistic sign-flip permutation test on the net (forward − backward) sequenceness against zero.

`params` requires: `MAX_LAG`, `tm`, `N_SIGN_PERMS`. Returns `xcorr_data[seg]` with `sequenceness`, `permutations`, `empirical_tm`, `skip`, `seq_net`, `perm_net`, `stat_info`, `tmap`, `threshold`, `threshold_data`.

**`run_group_comparison(xcorr_data_old, xcorr_data_young, params, segments)`**

Two-sample, two-sided label-switch permutation test comparing old vs. young net sequenceness. Returns `xcorr_data_diff[seg]` with `seq_diff`, `seq_old`, `seq_young`, `stat_info`, `tmap`, `surrogate_tmap`, `threshold`, `threshold_data`.

**`run_contrasts(xcorr_data, params, base_seg, comparison_segs)`**

Within-group, one-sided (`base_seg` > `comparison_seg`) label-switch permutation test, e.g. `cued_replay` vs. `seq_learn` / `preresting`. Returns `xcorr_contr[comp_seg]` with the same fields as `run_group_comparison`.

---

### freq.py

**`band_peak_power(psd_ch, freqs, fmin, fmax)`**

Power at the highest *local* peak (via `scipy.signal.find_peaks`) within `[fmin, fmax]`; falls back to the band maximum if the spectrum is monotone in that range.

**`run_freq_analysis(eeg_data, params)`**

Computes per-channel Welch PSD (default segment: `"cued_replay"`), flags outlier channels via MAD thresholding in the theta band, and identifies the channel with the highest in-band theta peak per participant.

`params` requires: `FMIN_PSD`, `FMAX_PSD`, `FMIN_BAND`, `FMAX_BAND`, `N_FFT_OSC`, `OUTLIER_THRESH_MAD`, optional `segment` (default `"cued_replay"`).

Returns a dict with `psd_data`, `psd_freqs`, `avg_power`, `peak_power`, `ch_outlier`, `max_ch_idx_peak`, `max_ch_peak_vals`.

---

### utils.py

Statistical and helper functions.

- **`chan_grid`** — 31-channel EEG name → topographic grid-index mapping, used by `plot_topo_cond`.
- **`std_error(data, axis=0, ddof=1)`** — standard error of the mean.
- **`sign_flip_permtest(data, n_permutations=1000, clust_thresh_pval=0.05, **kwargs)`** — cluster-based or max-statistic sign-flip permutation test for 1D/2D (or 3D temporal-generalization) group data against a chance level (`kwargs["chance_lev"]`). `kwargs`: `sided`, `chance_lev`, `test` (`"cluster"` or `"max"`), `add_info`, `time_info`.
- **`label_switch_permtest(data_a, data_b, n_permutations=1000, clust_thresh_pval=0.05, **kwargs)`** — two-sample permutation test via label switching (for independent-group comparisons where sign-flipping doesn't apply). Same return structure as `sign_flip_permtest`.

---

### plots.py

Visualization utilities for EEG analysis results.

```python
plot_data_quant(trialnum_loc, rest_dur, labels, ytick_names, **kwargs)
plot_cohens_kappa(kappa_values, scale="landis", **kwargs)
plot_correlation(x, y, **kwargs)
plot_cond(data, x, labels, **kwargs)
plot_chan_cond(data, x, channels, labels, **kwargs)
plot_topo_cond(data, x, channels, labels, **kwargs)  # interactive, click a sensor to zoom
plot_sliding_classifier(data, time, **kwargs)
plot_topo_class_pattern(data, time, epoch, **kwargs)
plot_generalizing_classifier(data, timex, timey, **kwargs)
plot_sequenceness(data, lags, perms, **kwargs)
plot_sequenceness_subjects(data, lags, perms, **kwargs)
add_significance(
    ax, stat_info, plot_type="area", times=None, timex=None, timey=None, alpha_thresh=0.05, **kwargs
)
```

`add_significance` is the shared significance-overlay function: it accepts cluster-format (`sign_flip_permtest`/`label_switch_permtest` with `test="cluster"`), max-format (`test="max"`), or the legacy pre-processed dict, and supports both 1D (`"area"`, `"lines"`, `"linepoints"`, `"points"`, `"bar"`) and 2D (`"alpha"`, `"contour"`, `"hatch"`) plot types.

## Example Usage

```python
from utils.imports import Things_Importer
from utils.decode import run_decode
from utils.xdecode import run_xdecode
from utils.xcorr import run_xcorr
from utils.plots import plot_sequenceness
from utils.paths import INFO_DIR, PREPROC_DIR

importer = Things_Importer(INFO_DIR, PREPROC_DIR)
participant_info = importer.get_participant_info()
eeg_data = importer.get_eeg_data(sample_mask=participant_info["participants_incl"])
sample_mask = range(len(participant_info["participants_incl"]))

params = {
    "Cs": [6] * len(participant_info["participants_incl"]),
    "solver": "liblinear",
    "penalty": "l1",
    "max_iter": 10000,
    "CV": 6,
    "N_SIGN_PERMS": 1000,
    "CLUST_THRESH_PVAL": 0.05,
    "training_timepoint": 0.4,
    "n_classes": 3,
    "MAX_LAG": 50,
    "tm": [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
}

classifier_data = run_decode(eeg_data, sample_mask, params)
decode_data = run_xdecode(eeg_data, sample_mask, params, segments=["resting", "cued_replay"])
xcorr_data = run_xcorr(decode_data, params, segments=["resting", "cued_replay"])

lags_ms = [10 * (i + 1) for i in range(params["MAX_LAG"])]
plot_sequenceness(
    xcorr_data["cued_replay"]["seq_net"], lags_ms, xcorr_data["cued_replay"]["perm_net"]
)
```
