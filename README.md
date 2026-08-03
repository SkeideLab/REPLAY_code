# REPLAY Code

Analysis pipeline for detecting sequential neural replay patterns in infant EEG data.

**Author:** Christopher Postzich
**License:** MIT (SkeideLab 2025)

## Overview

This project implements a complete EEG analysis workflow to detect memory replay during rest, cued reactivation, and sequence learning in infants. Two age groups are processed in parallel throughout the pipeline:

- **old**: 10–13 months
- **young**: 6–9 months

The pipeline includes:

- Participant filtering and EEG preprocessing (band-pass filtering, decimation)
- Time-resolved logistic-regression decoding of localizer stimuli (Apple, Chair, Face)
- Cross-temporal decoding of stimulus probabilities into resting/replay/learning segments
- Cross-correlation based sequenceness analysis (forward vs. backward replay)
- Sign-flip and label-switch permutation testing (within- and between-group)
- Welch PSD / individual theta-peak analysis
- Publication-quality visualizations, including a set of schematic methods figures

## Project Structure

```
REPLAY_code/
├── load_data.py               # Main pipeline: loads data, decodes, computes sequenceness & frequency stats
├── plot_results.py            # Main results figures (run after load_data.py, same session)
├── plot_supplement.py         # Supplementary figures (run after load_data.py, same session)
├── plot_methods.py            # Standalone schematic simulations for the methods figure
├── utils/                     # Utility package
│   ├── paths.py                # Central path configuration (edit before running)
│   ├── imports.py              # Things_Importer + get_additional_data
│   ├── decode.py                # Localizer sliding classifier (run_decode)
│   ├── CrossDecoding_MEEG.py   # Cross-temporal decoding estimator
│   ├── xdecode.py               # Cross-decoding across segments (run_xdecode)
│   ├── CrossCorrelation.py      # Cross-correlation sequenceness model
│   ├── xcorr.py                  # Sequenceness + permutation tests
│   ├── freq.py                    # Welch PSD / theta-peak analysis (run_freq_analysis)
│   ├── utils.py                    # chan_grid, std_error, permutation-test helpers
│   └── plots.py                     # Plotting utilities
├── additional_data/            # Participant metadata and generated trial summaries
│   ├── participants_info.xlsx    # Participant registry (input)
│   ├── trial_info_old.csv         # Per-participant trial counts, old group (generated)
│   └── trial_info_young.csv        # Per-participant trial counts, young group (generated)
├── LICENSE
└── pyproject.toml
```

See [utils/README.md](utils/README.md) and [additional_data/README.md](additional_data/README.md) for details on each module and data file.

## Installation

### Requirements

- Python >= 3.10
- MNE-Python (EEG data handling)
- NumPy, SciPy, Pandas
- Scikit-learn (classifiers, permutation tests)
- Scikit-image (cluster labeling for permutation tests)
- Matplotlib (visualization)
- tqdm (progress bars)
- python-calamine (fast `.xlsx` reading via `pandas.read_excel(engine="calamine")`)
- A Qt binding (e.g. PyQt6) for the interactive `qtagg` Matplotlib backend used by the `plot_*.py` scripts

### Setup

```bash
# Clone the repository
git clone https://github.com/SkeideLab/REPLAY.git
cd REPLAY/Analysis/REPLAY_code

# Install dependencies (using pip)
pip install -e .

# Or install dependencies directly
pip install numpy scipy pandas scikit-learn scikit-image mne tqdm matplotlib python-calamine pyqt6
```

### Configuration

Before running anything, edit the placeholder paths in [utils/paths.py](utils/paths.py):

| Constant | Purpose |
|----------|---------|
| `RAW_DIR` | Raw data root (behavioral coding / attention ratings) |
| `PREPROC_DIR` | Preprocessed EEG segment root (expects `Segments/<Localizer\|Resting\|CuedReplay\|LearnSequence\|PreResting>/`) |
| `GRAPHICS_DIR` | Figure output root — create `Results/` and `Paradigm/` subfolders before running the plotting scripts |

`INFO_DIR` (→ `additional_data/`) and `UTILS_DIR` are resolved automatically relative to the repo.

## Usage

The scripts are written to be run interactively (e.g. cell-by-cell in Spyder, a VS Code Interactive Window, or Jupyter) rather than as isolated subprocess calls: `plot_results.py` and `plot_supplement.py` reference variables (`eeg_data_old`, `classifier_data_young`, `xcorr_data_diff`, …) that only exist once `load_data.py` has been executed in the same session.

### 1. Load, decode, and analyze

```python
# Run in an interactive session — populates the workspace used by the plotting scripts
exec(open("load_data.py").read())
```

This script:
1. Loads `participants_info.xlsx` and splits participants into the **old** and **young** age groups
2. Loads and preprocesses EEG segments (localizer, resting, cued replay, sequence learning, pre-resting) per group
3. Computes per-participant descriptive stats (trial counts, resting duration, inter-rater Cohen's kappa) and writes `additional_data/trial_info_old.csv` / `trial_info_young.csv`
4. Trains a sliding logistic-regression classifier on localizer stimuli and tests it against chance with a sign-flip cluster permutation test
5. Cross-decodes localizer-trained classifiers onto each segment
6. Computes cross-correlation sequenceness (forward/backward/net) per segment, with sign-flip max-statistic permutation tests
7. Runs between-group (old vs. young) and within-group (`cued_replay` vs. `seq_learn` / `preresting`) label-switch permutation comparisons
8. Runs a Welch PSD / individual theta-peak analysis on the cued-replay segment

### 2. Generate figures

In the **same session**, after `load_data.py` has run:

```python
exec(open("plot_results.py").read())  # main figures
exec(open("plot_supplement.py").read())  # supplementary figures
```

`plot_methods.py` is independent of the loaded data (it only simulates schematic curves for the methods figure) and can be run on its own:

```python
exec(open("plot_methods.py").read())
```

### Using individual modules

```python
from utils.imports import Things_Importer
from utils.CrossCorrelation import CrossCorrelationSequenceness
from utils.xcorr import run_xcorr

importer = Things_Importer(info_dir=INFO_DIR, preproc_dir=PREPROC_DIR)
participant_info = importer.get_participant_info()
eeg_data = importer.get_eeg_data(sample_mask=participant_info["participants_incl"])

# Cross-correlation sequenceness for a single participant
xcorr_model = CrossCorrelationSequenceness(max_lag=50)
tm = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # forward: state0 -> state1 -> state2
sequenceness = xcorr_model.fit(stimulus_probs, classes, model_tm=tm)  # (3, max_lag): fwd, bwd, net
```

## Experimental Segments

| Segment | Description |
|---------|-------------|
| `localizer` | Training data with known stimulus labels (Apple, Chair, Face) |
| `resting` | Spontaneous activity for replay detection |
| `cued_replay` | Stimulus-cued reactivation during resting |
| `seq_learn` | Activity during sequence presentation |
| `preresting` | Baseline before learning (merged `PreResting` + `PreResting_Break` epochs) |

## Key Methods

### Cross-Temporal Decoding
`CrossDecoding_MEEG` (scikit-learn compatible) trains a classifier at one localizer timepoint per participant and applies it across all timepoints of a target segment, yielding per-class stimulus-probability time series.

### Cross-Correlation Sequenceness
`CrossCorrelationSequenceness` computes the lagged Pearson cross-correlation between stimulus-probability time series and projects it onto a transition matrix to obtain forward, backward, and net (forward − backward) sequenceness — the cross-correlation analogue of TDLM-style sequenceness analysis.

### Permutation Testing
- `sign_flip_permtest`: within-group cluster- or max-statistic sign-flip permutation test (used for classifier-vs-chance and sequenceness-vs-zero tests).
- `label_switch_permtest`: between/within-group two-sample permutation test via label switching (used for old-vs-young and within-group segment contrasts).

## Output

Results are saved to:
- `additional_data/trial_info_old.csv`, `additional_data/trial_info_young.csv` — per-participant trial summaries
- `<GRAPHICS_DIR>/Results/` — main and supplementary figures (PNG/SVG)
- `<GRAPHICS_DIR>/Paradigm/` — schematic methods figures (SVG)

## Citation

If you use this code, please cite the associated preprint [a link](https://doi.org/10.1101/2025.06.12.659246).

## License

MIT License - see [LICENSE](LICENSE) for details.

## References
Christopher M. Postzich, Johanna Finnemann, Michael A. Skeide
bioRxiv 2025.06.12.659246; doi: https://doi.org/10.1101/2025.06.12.659246
