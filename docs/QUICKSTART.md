# Quick Start Guide

## 5-Minute Setup

### 1. Download the Dataset

```bash
# Download EEGMMIDB from PhysioNet
wget -r -N -c -np https://physionet.org/files/eegmmidb/1.0.0/

# Or download via web browser:
# https://physionet.org/content/eegmmidb/1.0.0/
```

### 2. Install Dependencies

```bash
# Create environment
conda create -n eeg_analysis python=3.10
conda activate eeg_analysis

# Install packages
pip install mne numpy pandas matplotlib scipy openpyxl
pip install mne-icalabel autoreject
```

### 3. Configure Paths

Edit these scripts to point to your data:

**EDIH_Preprocessing_v0_1.py** (line ~176-180):
```python
data_root = Path("path/to/eeg-motor-movementimagery-dataset-1.0.0/files")
clean_root = Path("path/to/cleaned-dataset")
```

**All analysis scripts** (around line ~38-42):
```python
clean_root = Path("path/to/cleaned-dataset")
```

### 4. Run Preprocessing

```bash
# Start with a small test (3 subjects)
# Edit subject_ids range in EDIH_Preprocessing_v0_1.py (line ~182)
subject_ids = range(1, 4)  # S001, S002, S003

# Run
python EDIH_Preprocessing_v0_1.py
```

**Expected time**: ~15-20 minutes for 3 subjects (14 runs each)

### 5. Run Analysis

```bash
# Screen responders
python Task3_responder_screen_S001_S109.py

# Compare channel sets
python Phase1_Step1A_ExtendedChannel_ERD_Discriminability_Task3.py

# Multi-band analysis
python Phase1_Step1B_MultiBand_Discriminability_Task3.py

# Generate plots
python Phase1_Step1B_RepresentativeSubjects_Plots.py
```

---

## Understanding Your Results

### 1. Responder Screening Output

**File**: `task3_responder_screen_S001_S109.csv`

Look for:
```python
import pandas as pd
df = pd.read_csv("task3_responder_screen_S001_S109.csv")

# Top 10 dual responders
top10 = df[df["dual_responder"] == True].sort_values("dual_total_score", ascending=False).head(10)
print(top10[["subject", "dual_total_score", "mu_best_channel", "beta_best_channel"]])
```

**What it means**:
- `dual_responder = True`: Subject shows strong ERD in BOTH mu and beta
- High `dual_total_score`: Better for BCI applications
- `mu_best_channel`, `beta_best_channel`: Where to look for activity

### 2. Channel Set Comparison

**File**: `Phase1_Step1A_Task3_ERD_discriminability_min_vs_ext.csv`

```python
df = pd.read_csv("Phase1_Step1A_Task3_ERD_discriminability_min_vs_ext.csv")

# Did extended channels help?
improved = df[df["delta_combo"] > 0]
print(f"Extended improved: {len(improved)}/{len(df)} subjects ({100*len(improved)/len(df):.1f}%)")

# Top improvers
top = df.sort_values("delta_combo", ascending=False).head(10)
print(top[["subject", "min_combo_score", "ext_combo_score", "delta_combo"]])
```

**What it means**:
- `delta_combo > 0`: Extended channels (FC/C/CP grid) helped this subject
- Large `delta_combo`: Strong evidence for using broader electrode montage

### 3. Multi-Band Analysis

**File**: `Phase1_Step1B_Task3_band_discriminability_motor_strip.csv`

```python
df = pd.read_csv("Phase1_Step1B_Task3_band_discriminability_motor_strip.csv")

# Which bands dominate?
print(df["best_band"].value_counts())

# Mean improvement over mu_beta baseline
print(f"Mean improvement: {df['best_minus_baseline'].mean():.3f}")

# Subjects with narrow beta advantage
narrow_beta = df[df["best_band"].isin(["beta_low", "beta_high"])]
print(f"Narrow beta optimal: {len(narrow_beta)}/{len(df)} subjects")
```

**What it means**:
- `best_band`: Frequency range with highest T1 vs T2 separation
- Common findings:
  - ~30% show mu dominance (8-13 Hz)
  - ~25% show beta_low dominance (13-20 Hz)
  - ~20% show beta_high dominance (20-30 Hz)
  - ~15% show theta dominance (4-8 Hz)

### 4. Plots

**Directory**: `Phase1_Step1B_RepresentativeSubjects_Plots_figs/`

Check:
- **Plot 1 (TFR)**: Look for blue (negative, ERD) regions in task window (>0 s)
- **Plot 2 (Bar chart)**: Tallest bar = optimal band for this subject
- **Plot 3 (PSD)**: T1 and T2 should have different power in optimal band

---

## Typical Results

### Example: Strong Mu Responder (S049)

```
Best band: mu (8-13 Hz)
Best maxdiff: 28.5%
Baseline (mu_beta): 22.1%
Improvement: +6.4%

Interpretation:
- Strong ERD in mu band during motor execution
- 6.4% better discrimination using narrow mu vs. broad mu+beta
- Good candidate for mu-rhythm based BCI
```

### Example: Beta_low Responder (S016)

```
Best band: beta_low (13-20 Hz)
Best maxdiff: 19.3%
Baseline (mu_beta): 15.8%
Improvement: +3.5%

Interpretation:
- Optimal response in lower beta (13-20 Hz)
- May require band-specific filtering for classification
- Could benefit from adaptive frequency selection
```

### Example: Non-Responder (S058)

```
Best band: mu
Best maxdiff: 7.2%
Baseline (mu_beta): 6.8%
Improvement: +0.4%

Interpretation:
- Weak discrimination across all bands
- May have high inter-trial variability
- Consider excluding from BCI training set
```

---

## Common Patterns

### Good Data Quality Indicators

✅ **Preprocessing**:
- <5% bad channels per subject
- ICLabel excludes 3-10 components (eye, muscle, heart)
- AutoReject drops <20% of epochs

✅ **Analysis**:
- Dual responder rate: 40-60% of subjects
- Mean ERD: -15% to -25% in optimal band
- Negative ERD consistency: >60% of epochs

### Red Flags

⚠️ **Preprocessing**:
- >20% bad channels → Poor electrode impedance
- ICLabel excludes <2 or >15 components → ICA issues
- AutoReject drops >50% epochs → Noisy data

⚠️ **Analysis**:
- Dual responder rate: <20% → Check preprocessing
- Mean ERD: >-5% → Insufficient motor response
- High positive ERD: Possible artifacts or wrong task

---

## Optimization Tips

### 1. Faster Preprocessing

```python
# Reduce ICA components (faster fitting)
n_components = 0.95  # Instead of 0.99

# Skip AutoReject for initial runs
HAVE_AUTOREJECT = False
```

### 2. Focus on Top Subjects

```python
# In Step 1B (line ~82)
TOP_N = 20  # Instead of 40, focus on best responders
```

### 3. Parallel Processing

```bash
# Split into batches
# Terminal 1:
python preprocess_batch_1.py  # S001-S027

# Terminal 2:
python preprocess_batch_2.py  # S028-S054

# Terminal 3:
python preprocess_batch_3.py  # S055-S081

# Terminal 4:
python preprocess_batch_4.py  # S082-S109
```

### 4. GPU Acceleration (if available)

```python
# Enable MNE CUDA (experimental)
import os
os.environ["MNE_USE_CUDA"] = "true"
```

---

## Validation Checklist

Before proceeding to Phase 2 (CSP classification):

- [ ] All subjects preprocessed successfully
- [ ] Bad channel log reviewed (no systematic issues)
- [ ] At least 30 subjects identified as dual responders
- [ ] Multi-band analysis shows band diversity (not all mu or all beta)
- [ ] QC plots confirm ERD patterns visually
- [ ] Representative subject plots generated
- [ ] CSV files saved and readable

---

## Getting Help

### Debugging

```python
# Enable verbose output in any script
# Change verbose="ERROR" to verbose="INFO"
ep = mne.read_epochs(epo_path, preload=True, verbose="INFO")

# Check epoch counts
print(f"Epochs: {len(epochs)}, Events: {epochs.event_id}")

# Inspect specific subject
raw = mne.io.read_raw_fif("cleaned-dataset/S001/S001R05-cleaned_raw.fif")
raw.plot()  # Interactive viewer
```

### Common Questions

**Q: How long does full preprocessing take?**
A: ~6-8 hours for all 109 subjects (single-threaded, standard laptop)

**Q: How much disk space needed?**
A: ~50-60 GB for all subjects (raw + epochs + proxy ECG)

**Q: Can I skip baseline runs (R01, R02)?**
A: Yes, the pipeline skips epoching for R01/R02 by default (line ~205 in EDIH_Preprocessing_v0_1.py)

**Q: What if ICLabel is not available?**
A: Pipeline falls back to heuristic EOG detection. Results will be similar but proxy ECG unavailable.

**Q: Should I use minimal or extended channels?**
A: Run Step 1A first to determine subject-specifically. Most subjects benefit from extended.

---

## Next Steps

Once you've completed Phase 1:

1. **Review results**: Identify top 20-30 subjects for BCI development
2. **Phase 2**: Implement CSP+LDA classification (not included in this repository)
3. **Validate findings**: Confirm band preferences with actual decoding accuracy
4. **Publish**: Use visualizations and CSV summaries in papers/reports

---

## Useful Resources

- **MNE Tutorial**: https://mne.tools/stable/auto_tutorials/
- **EEGMMIDB Details**: https://physionet.org/content/eegmmidb/1.0.0/
- **ERD/ERS Review**: Pfurtscheller & Lopes da Silva (1999)
- **BCI Competition**: http://www.bbci.de/competition/

---

**Pro tip**: Start with the top 10 dual responders for BCI prototype development. They have the most reliable and strong ERD patterns.
