# Additional Data

Data files required for the REPLAY analysis pipeline.

## Files

### participant_info.xlsx

Participant metadata spreadsheet containing demographic and inclusion information.

**Columns:**
| Column | Description |
|--------|-------------|
| `participant_id` | Unique participant identifier |
| `age_months` | Age at testing in months |
| `gender` | Participant gender (M/F) |
| `included` | Inclusion status (1=included, 0=excluded) |
| `exclusion_reason` | Reason for exclusion if applicable |
| `n_trials_*` | Number of valid trials per segment |
| `resting_duration` | Duration of resting state recording |

**Usage:**
```python
import pandas as pd

# Load participant info
pinfo = pd.read_excel('participant_info.xlsx')

# Filter included participants
included = pinfo[pinfo['included'] == 1]

# Filter by age range
age_filtered = pinfo[(pinfo['age_months'] >= 6) & (pinfo['age_months'] <= 12)]
```

---

### trial_info.csv

Spreadsheet containing demographic and trial inclusion information.

**Columns:**
| Column | Description |
|--------|-------------|
| `Participant` | Unique participant identifier |
| `Age` | Age at testing in months |
| `Gender` | Participant gender (M/F) |
| `Localizer_Trials` | Amount of Localizer Trials used for analyses. |
| `Localier_Trials_Percent` | Percent of Localizer Trials used for analyses. |
| `Sequence_Presentations` |  Amount of Sequence Presentations used for analyses. |
| `Sequence_Presentations_Percent` | Percent of Sequence Presentations used for analyses. |

### clusterdepth_pvals.npz

Pre-computed cluster-depth-based permutation test results for cued replay analysis. Permutation tests were computed in R using the permuco package.

**Contents:**
- `cluster_pval1`: P-values for stimulus category 1 (Apple)
- `cluster_pval2`: P-values for stimulus category 2 (Chair)
- `cluster_pval3`: P-values for stimulus category 3 (Face)

**Data Structure:**
Each array has shape `(n_timepoints, 3)` containing:
- Column 0: Test statistic (t-value)
- Column 1: Corrected p-value (cluster-based)
- Column 2: Significance mask (1=significant, 0=not significant)

**Usage:**
```python
import numpy as np

# Load pre-computed statistics
data = np.load('clusterdepth_pvals.npz')

# Access p-values for each stimulus
apple_stats = data['cluster_pval1']
chair_stats = data['cluster_pval2']
face_stats = data['cluster_pval3']

# Get significance masks
apple_sig = apple_stats[:, 2].astype(bool)
chair_sig = chair_stats[:, 2].astype(bool)
face_sig = face_stats[:, 2].astype(bool)

# Get corrected p-values
apple_pvals = apple_stats[:, 1]
```

## Data Location

The EEG data files referenced by `utils/imports.py` are stored separately and should be placed in the expected directory structure:

```
REPLAY/
├── Data/
│   ├── Preprocessed/
│   │   ├── sub-001/
│   │   │   ├── localizer_epo.fif
│   │   │   ├── resting_epo.fif
│   │   │   └── ...
│   │   └── sub-002/
│   │       └── ...
│   └── Behavioral/
│       └── responses.csv
└── Analysis/
    └── REPLAY_code/
        └── additional_data/  (this folder)
```

## Adding New Data

When adding new data files:

1. Use descriptive filenames indicating content and format
2. For NumPy arrays, prefer `.npz` (compressed) over `.npy`
3. For tabular data, use `.xlsx` or `.csv`
4. Document the file structure in this README
5. Include units and data types where applicable
