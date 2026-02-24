# Data Setup

## SEED-DV Dataset

The SEED-DV dataset is not included in this repository. Please follow the steps below to obtain it.

### Download

1. Visit the official SEED-DV release page (linked from the EEG2Video repo):
   https://github.com/XuanhaoLiu/EEG2Video

2. Fill out the data access form if required by the dataset maintainers.

3. Download and extract the dataset to this `data/` directory (or update
   `configs/config.yaml` to point to your local copy).

### Expected Structure

After downloading, the directory should look like:

```
data/
├── README.md               ← this file
├── raw/
│   ├── sub01/
│   │   ├── block1.mat
│   │   └── ...
│   └── sub20/
├── labels/
│   ├── concept_labels.csv
│   ├── color_labels.csv
│   └── motion_labels.csv
└── videos/                 ← optional; needed for reconstruction (E3)
    └── ...
```

### Notes

- EEG data is stored as 62-channel, 2-second epochs aligned to video onset.
- 20 subjects × 7 blocks × 200 trials per block = 1,400 trials per subject.
- The run structure: every 5 consecutive trials share the same concept category.
- Raw data should **not** be committed to Git. The `data/raw/` and `data/videos/`
  directories are listed in `.gitignore`.
