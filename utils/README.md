# Utils Module

Utility functions and classes for the REPLAY EEG analysis pipeline.

## Modules

### imports.py

Data import functionality for the REPLAY-Things experiment.

**Main Function:**
```python
import_preproc_data_replay_things(
    segment=['localizer', 'resting'],
    resample_to=125,
    age_range=(6, 12),
    require_all_segments=True
)
```

**Parameters:**
- `segment`: List of experimental segments to load
- `resample_to`: Target sampling rate in Hz
- `age_range`: Tuple of (min_age, max_age) in months for filtering
- `require_all_segments`: Only include participants with all requested segments

**Returns:**
- `participant_info`: Dict with participant metadata
- `eeg_data`: Dict with MNE Epochs objects per segment
- `behavioral_data`: Dict with behavioral responses

---

### utils.py

Statistical and helper functions.

**Functions:**

```python
std_error(data, axis=0)
```
Computes standard error of the mean.

```python
sign_flip_permtest(data, n_permutations=1000, cluster_threshold=0.05)
```
Cluster-based sign-flip permutation test for 1D or 2D data. Returns t-statistics, p-values, and significant cluster masks.

```python
hotelling_t2_test(X, Y)
```
Hotelling's T-squared test for functional data analysis.

**Data:**
- `chan_grid`: 32-channel EEG position mapping for topographic plots

---

### TDLM.py

Temporally Delayed Linear Model for detecting sequential replay.

**Classes:**

```python
class TDLM:
    def __init__(self, max_lag=600, bin_lag=8, permutation_method='circshift'):
        ...

    def fit(self, classifier_probs, transition_matrix):
        """Compute sequenceness measures."""
        ...

    def permutations(self, n_permutations=1000):
        """Generate null distribution."""
        ...

    def replay_onsets(self, threshold=2.0):
        """Identify replay event timing."""
        ...
```

**Functions:**

```python
create_transition_matrix(n_states, hypothesis='forward')
```
Create transition matrices for hypothesis testing.

**Supported hypotheses:**
- `'forward'`: A -> B -> C
- `'backward'`: C -> B -> A
- `'circular'`: A -> B -> C -> A
- `'symmetric'`: Forward + Backward

---

### CrossDecoding_MEEG.py

Cross-temporal decoding classifier (scikit-learn compatible).

```python
class CrossDecoding_MEEG:
    def __init__(self, base_estimator=None, training_timepoint=0,
                 baseline_timepoint=None, include_zero=True):
        ...

    def fit(self, X, y):
        """Train classifier at specified timepoint(s)."""
        ...

    def predict_proba(self, X):
        """Predict probabilities across all timepoints."""
        ...
```

**Parameters:**
- `base_estimator`: Classifier to use (default: LogisticRegression)
- `training_timepoint`: Index or list of indices for training
- `baseline_timepoint`: Optional baseline for augmentation
- `include_zero`: Include zero-class predictions

---

### plots.py

Visualization utilities for EEG analysis results.

**Functions:**

```python
plot_sliding_classifier(times, accuracy, chance_level=0.33, ci=None)
```
Plot time-resolved decoding accuracy.

```python
plot_topo_class_pattern(pattern, info, title='')
```
Plot topographic patterns from classifier weights.

```python
plot_sequenceness(lags, sequenceness, permutation_band=None)
```
Plot sequenceness across time lags with significance bands.

```python
add_signif_timepts(ax, times, mask, y_position=-0.1)
```
Add significance markers to time series plots.

```python
plot_cond(data, conditions, colors=None)
```
Compare conditions with error bars and statistics.

## Example Usage

```python
from utils.imports import import_preproc_data_replay_things
from utils.TDLM import TDLM, create_transition_matrix
from utils.CrossDecoding_MEEG import CrossDecoding_MEEG
from utils.utils import sign_flip_permtest
from utils.plots import plot_sequenceness

# Load data
pinfo, eeg, behav = import_preproc_data_replay_things(
    segment=['localizer', 'resting']
)

# Train decoder
decoder = CrossDecoding_MEEG(training_timepoint=50)
decoder.fit(X_train, y_train)
probs = decoder.predict_proba(X_test)

# Analyze sequenceness
trans = create_transition_matrix(3, 'forward')
tdlm = TDLM(max_lag=600)
seq = tdlm.fit(probs, trans)

# Statistical test
t_stat, pvals, mask = sign_flip_permtest(seq)

# Visualize
plot_sequenceness(tdlm.lags, seq, permutation_band=tdlm.permutations(1000))
```
