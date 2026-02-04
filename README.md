# REPLAY Code

Analysis pipeline for detecting sequential neural replay patterns in infant EEG data.

**Author:** Christopher Postzich
**License:** MIT (SkeideLab 2025)

## Overview

This project implements a complete EEG analysis workflow to detect memory replay during rest and learning in infants. The pipeline includes:

- Loading and preprocessing EEG data
- Training logistic regression classifiers on localizer stimuli
- Cross-temporal decoding across experimental segments
- Temporally Delayed Linear Model (TDLM) analysis for sequenceness detection
- Statistical inference via permutation testing
- Publication-quality visualizations

## Project Structure

```
REPLAY_code/
├── load_transform_data.py    # Main analysis pipeline
├── plot_results.py           # Results visualization
├── utils/                    # Utility modules
│   ├── imports.py            # Data import functions
│   ├── utils.py              # Statistical helpers
│   ├── TDLM.py               # Temporal sequenceness analysis
│   ├── CrossDecoding_MEEG.py # Cross-temporal decoding
│   └── plots.py              # Plotting utilities
├── additional_data/          # Data files
│   ├── participant_info.xlsx # Participant metadata
│   └── clusterdepth_pvals.npz# Pre-computed statistics
├── LICENSE
└── pyproject.toml
```

## Installation

### Requirements

- Python >= 3.10
- MNE-Python (EEG data handling)
- NumPy, SciPy, Pandas
- Scikit-learn (classifiers)
- Matplotlib, Seaborn (visualization)
- Openpyxl (Excel file reading)

### Setup

```bash
# Clone the repository
git clone https://github.com/SkeideLab/REPLAY.git
cd REPLAY/Analysis/REPLAY_code

# Install dependencies (using pip)
pip install -e .

# Or install dependencies directly
pip install mne numpy scipy pandas scikit-learn matplotlib seaborn openpyxl
```

## Usage

### Main Analysis Pipeline

The primary analysis is performed via `load_transform_data.py`:

```python
# Run the full analysis pipeline
python load_transform_data.py
```

This script:
1. Loads preprocessed EEG segments (localizer, resting, cued replay, sequence learning)
2. Trains classifiers on localizer stimuli (Apple, Chair, Face)
3. Applies cross-temporal decoding
4. Computes TDLM sequenceness measures
5. Saves results for visualization

### Generating Figures

```python
# Generate publication figures
python plot_results.py
```

Creates visualizations including:
- Time-resolved decoding curves
- Spatial patterns and topographies
- Sequenceness time-lag plots
- Transition matrix comparisons

### Using Individual Modules

```python
from utils.imports import import_preproc_data_replay_things
from utils.TDLM import TDLM, create_transition_matrix
from utils.CrossDecoding_MEEG import CrossDecoding_MEEG

# Load data
participant_info, eeg_data, behavioral_data = import_preproc_data_replay_things(
    segment=['localizer', 'resting'],
    resample_to=100
)

# Create transition matrix for hypothesis testing
trans_matrix = create_transition_matrix(n_states=3, hypothesis='forward')

# Run TDLM analysis
tdlm = TDLM(max_lag=600, bin_lag=8)
sequenceness = tdlm.fit(classifier_probs, trans_matrix)
```

## Experimental Segments

| Segment | Description |
|---------|-------------|
| `localizer` | Training data with known stimulus labels |
| `resting` | Spontaneous activity for replay detection |
| `pre_resting` | Baseline before learning |
| `cued_replay` | Stimulus-cued reactivation during resting |
| `seq_learn` | Activity during sequence presentation |

## Key Methods

### Cross-Temporal Decoding
Trains classifiers at specific timepoints and tests across all timepoints to identify temporal generalization patterns.

### TDLM (Temporally Delayed Linear Model)
Detects sequential reactivation by modeling lagged relationships between classifier outputs, identifying forward and backward replay.

### Permutation Testing
Cluster-based sign-flip permutation tests for statistical inference on sequenceness measures.

## Output

Results are saved to:
- `../Results/` - Processed data arrays (NPZ format)
- `../../Graphics/Results/` - Figures (SVG/PNG format)

## Citation

If you use this code, please cite the associated preprint [a link](https://doi.org/10.1101/2025.06.12.659246).

## License

MIT License - see [LICENSE](LICENSE) for details.

## References
Christopher M. Postzich, Johanna Finnemann, Michael A. Skeide
bioRxiv 2025.06.12.659246; doi: https://doi.org/10.1101/2025.06.12.659246 