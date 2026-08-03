# Additional Data

Participant metadata and pipeline-generated trial summaries for the REPLAY analysis.

## Files

### participants_info.xlsx

Master participant registry — one row per recording session. Read by `Things_Importer.get_participant_info()` (`utils/imports.py`) via `pandas.read_excel(engine="calamine", decimal=",")`.

**Columns:**
| Column | Description |
|--------|-------------|
| `Participant` | Unique participant code (e.g. `BabyPilot12`) |
| `Baby` | `1` if an infant participant, `0` for adult/pilot test recordings (excluded by default via `baby_only`) |
| `Sub_Code` | Internal lab subject code |
| `Date` | Session/testing date |
| `Time` | Session start time |
| `Cap_Size` | EEG cap size used |
| `Birthday` | Participant date of birth |
| `Gender` | `m` / `f` |
| `Video_useable` | `1`/`0` — whether behavioral video coding is usable |
| `Video_Notes` | Free-text notes on the video coding |
| `Localizer_usable` | `1`/`0` — whether the localizer segment passed QC |
| `Sequence_usable` | `1`/`0` — whether the sequence-learning segment passed QC |
| `Signal_Quality` | Categorical EEG signal-quality rating |
| `Include` | `1`/`0` master inclusion flag |
| `EOG` | `1`/`0` — whether an EOG channel is usable/available |
| `General_Notes` | Free-text notes |

Age in months is **not** stored directly — it is derived on the fly from `Birthday` and `Date` (leap-year corrected) inside `get_participant_info()`, then used to split participants into the **old** (~10–13 months) and **young** (~6–9 months) groups, each with independently configurable filters (`baby_only`, `include_only`, `min_age_months`, `max_age_months`, `require_localizer`, `require_sequence`).

**Usage:**
```python
from utils.imports import Things_Importer
from utils.paths import INFO_DIR, PREPROC_DIR

importer = Things_Importer(INFO_DIR, PREPROC_DIR)

# Old group (defaults: baby_only=True, include_only=True, 10-13 months, localizer+sequence required)
participant_info_old = importer.get_participant_info()

# Young group (override only the age range; other filters keep their defaults)
participant_info_young = importer.get_participant_info(
    participant_filters={"min_age_months": 5, "max_age_months": 10}
)
```

---

### trial_info_old.csv / trial_info_young.csv

**Generated outputs**, one per age group, written by `get_additional_data()` (`utils/imports.py`) each time `load_data.py` runs. Checked into the repo as a snapshot of the current dataset — not required as pipeline input.

**Columns:**
| Column | Description |
|--------|-------------|
| `Participant` | Sequential index (`1`, `2`, …) among *included* participants in this group — **not** the `Participant` code from `participants_info.xlsx` |
| `Age` | Age at testing in months |
| `Gender` | `m` / `f` |
| `Localizer_Trials` | Number of localizer trials retained after preprocessing/rejection |
| `Localizer_Trials_Percent` | `Localizer_Trials / 540` (540 = trials presented) |
| `Sequence_Presentations` | Number of retained sequence-learning presentations |
| `Sequence_Presentations_Percent` | `Sequence_Presentations / 100` (100 = presentations shown) |

**Usage:**
```python
import pandas as pd

trial_info_old = pd.read_csv("additional_data/trial_info_old.csv")
trial_info_young = pd.read_csv("additional_data/trial_info_young.csv")
```

## Data Location

The preprocessed EEG segment files and raw attention-rating files referenced by `utils/imports.py` are stored separately (not in this repo) and must be placed under the `RAW_DIR` / `PREPROC_DIR` paths configured in `utils/paths.py`:

```
<PREPROC_DIR>/
└── Segments/
    ├── Localizer/<Participant>_Epochs.fif
    ├── Resting/<Participant>_Epochs.fif
    ├── CuedReplay/<Participant>_Epochs.fif
    ├── LearnSequence/<Participant>_Epochs.fif
    └── PreResting/
        ├── <Participant>_Epochs.fif
        └── <Participant>_Break_Epochs.fif      # merged with the file above into "preresting"

<RAW_DIR>/
└── ratings/
    ├── rater1/Template_Localizer_<Participant>.csv     # ";" separated
    └── rater2/Template_Localizer_<Participant>.xlsx     # read via engine="calamine"
```
