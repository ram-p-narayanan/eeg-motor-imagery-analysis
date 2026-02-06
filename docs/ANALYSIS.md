# Analysis Pipeline Documentation

## Overview

The analysis pipeline validates Event-Related Desynchronization (ERD) patterns and identifies optimal frequency bands for motor imagery classification. All analyses are **pre-CSP** (Common Spatial Patterns), focusing on ERD discriminability rather than decoding accuracy.

## Pipeline Structure

```
Phase 1: Pre-CSP ERD Validation
│
├── Responder Screening
│   └── Task3_responder_screen_S001_S109.py
│
├── Step 1A: Channel Set Comparison
│   └── Phase1_Step1A_ExtendedChannel_ERD_Discriminability_Task3.py
│
├── Step 1B: Multi-Band Analysis
│   └── Phase1_Step1B_MultiBand_Discriminability_Task3.py
│
└── Visualization
    ├── Phase1_Step1B_RepresentativeSubjects_Plots.py
    └── Compare_S049_vs_S016_vs_S058_QCPlots.py
```

---

## 1. Responder Screening

**Script**: `Task3_responder_screen_S001_S109.py`

### Purpose

Identify "dual responders" - subjects showing strong ERD in both mu (8-13 Hz) and beta (13-30 Hz) bands during motor execution.

### Method

1. Load Task 3 epochs (runs R05, R09, R13)
2. Pick motor channels: C3, Cz, C4 (optionally C1, C2)
3. Compute ERD% for mu and beta bands
4. For each condition (fists T1, feet T2):
   - Calculate mean ERD per channel
   - Calculate % of epochs with negative ERD (desynchronization)
5. Score subjects on best channel performance

### Thresholds

```python
TH_MEAN = -10.0      # Mean ERD must be ≤ -10%
TH_NEGPCT = 60.0     # ≥60% of epochs show ERD
```

**Dual responder**: Meets thresholds in BOTH mu AND beta for at least one channel

### Output

**CSV**: `task3_responder_screen_S001_S109.csv`

Key columns:
- `subject`: Subject ID
- `dual_responder`: Boolean (True if meets criteria)
- `dual_total_score`: Combined mu + beta score
- `mu_best_channel`, `beta_best_channel`: Best performing channels
- `mu_best_score`, `beta_best_score`: Channel-specific scores
- Per-channel metrics: `mu_mean_best_C3`, `mu_negpct_best_C3`, etc.

### Usage

```python
# Run screening
python Task3_responder_screen_S001_S109.py

# Review top responders
import pandas as pd
df = pd.read_csv("task3_responder_screen_S001_S109.csv")
top10 = df.sort_values("dual_total_score", ascending=False).head(10)
print(top10[["subject", "dual_responder", "dual_total_score"]])
```

### Customization

```python
# Include more channels (line ~67)
picks = ["C3", "Cz", "C4", "C1", "C2", "C5", "C6"]

# Adjust thresholds (lines ~76-77)
TH_MEAN = -15.0      # Stricter
TH_NEGPCT = 70.0

# Change subject range (line ~61)
subjects = [f"S{i:03d}" for i in range(1, 50)]  # Only S001-S049
```

---

## 2. Step 1A: Channel Set Comparison

**Script**: `Phase1_Step1A_ExtendedChannel_ERD_Discriminability_Task3.py`

### Purpose

Determine whether extended sensorimotor channels improve Task 3 (fists vs feet) discriminability compared to minimal motor ROI.

### Channel Sets

1. **Minimal** (3 channels):
   ```python
   picks_minimal = ["C3", "Cz", "C4"]
   ```

2. **Extended** (up to 21 channels):
   ```python
   picks_extended = [
       # Left
       "FC5", "FC3", "FC1",
       "C5", "C3", "C1",
       "CP5", "CP3", "CP1",
       # Midline
       "FCz", "Cz", "CPz",
       # Right
       "FC2", "FC4", "FC6",
       "C2", "C4", "C6",
       "CP2", "CP4", "CP6"
   ]
   ```

### Analysis Windows

```python
baseline_win = (-0.5, 0.0)   # Pre-stimulus baseline
task_win = (0.5, 1.5)        # Active movement window
```

### Discriminability Metrics

For each band (mu: 8-13 Hz, beta: 13-30 Hz):

1. **ERD%** per epoch/channel:
   ```
   ERD% = (P_task - P_base) / P_base × 100
   ```

2. **Discriminability**:
   - `maxdiff`: max(|mean_ERD_T1 - mean_ERD_T2|) across channels
   - `meandiff`: mean(|diff|) across channels

3. **Combo score**: Average of mu_maxdiff + beta_maxdiff

### Output

**CSV**: `Phase1_Step1A_Task3_ERD_discriminability_min_vs_ext.csv`

Key columns:
- Minimal set: `min_n_ch`, `min_mu_maxdiff`, `min_beta_maxdiff`, `min_combo_score`
- Extended set: `ext_n_ch`, `ext_mu_maxdiff`, `ext_beta_maxdiff`, `ext_combo_score`
- Improvement: `delta_combo`, `delta_mu_maxdiff`, `delta_beta_maxdiff`

### Interpretation

```python
# Load results
df = pd.read_csv("Phase1_Step1A_Task3_ERD_discriminability_min_vs_ext.csv")

# Subjects where extended helped
improved = df[df["delta_combo"] > 0]
print(f"Extended improved: {len(improved)}/{len(df)} subjects")

# Top improvers
top = df.sort_values("delta_combo", ascending=False).head(15)
```

### Configuration

```python
# Require minimum channels for extended set (line ~69)
MIN_CH_FOR_EXT = 3

# Change bands (lines ~54-55)
band_mu = (8.0, 13.0)
band_beta = (13.0, 30.0)
```

---

## 3. Step 1B: Multi-Band Analysis

**Script**: `Phase1_Step1B_MultiBand_Discriminability_Task3.py`

### Purpose

Test whether narrower frequency sub-bands discriminate fists vs feet better than broad mu+beta.

### Frequency Bands

```python
bands_to_test = {
    "theta": (4.0, 8.0),
    "mu": (8.0, 13.0),
    "beta_low": (13.0, 20.0),
    "beta_high": (20.0, 30.0),
    "beta": (13.0, 30.0),
    "mu_beta": (8.0, 30.0),      # Baseline reference
}
```

### Channel Set

**Motor strip ROI** (7 channels):
```python
picks_motor_strip = ["C5", "C3", "C1", "Cz", "C2", "C4", "C6"]
```

### Subject Selection

1. **Preferred**: Top N subjects from Step 1A results
   ```python
   TOP_N = 40  # Default: top 40 by ext_combo_score
   ```

2. **Fallback**: All S001-S109 if Step 1A file not found

### Output

**CSV**: `Phase1_Step1B_Task3_band_discriminability_motor_strip.csv`

Key columns:
- `best_band`: Highest discriminability band for this subject
- `best_maxdiff`: Discriminability of best band
- `baseline_band`: "mu_beta" (8-30 Hz baseline)
- `baseline_maxdiff`: Baseline discriminability
- `best_minus_baseline`: Improvement over baseline
- `best_over_baseline_pct`: Percentage improvement
- Per-band metrics: `theta_maxdiff`, `mu_maxdiff`, etc.

### Interpretation

```python
# Load results
df = pd.read_csv("Phase1_Step1B_Task3_band_discriminability_motor_strip.csv")

# Band dominance distribution
win_counts = df["best_band"].value_counts()
print("Best-band counts:")
print(win_counts)

# Subjects benefiting from narrow beta
narrow_beta = df[df["best_band"].isin(["beta_low", "beta_high"])]
print(f"Narrow beta optimal: {len(narrow_beta)}/{len(df)} subjects")

# Mean improvement over baseline
print(f"Mean improvement: {df['best_minus_baseline'].mean():.3f}")
```

### Expected Findings

Based on typical motor imagery data:
- ~25-35% show **mu dominance** (8-13 Hz)
- ~20-30% show **beta_low dominance** (13-20 Hz)
- ~15-25% show **beta_high dominance** (20-30 Hz)
- ~10-20% show **theta dominance** (4-8 Hz, rare but possible)

---

## 4. Visualization: Representative Subjects

**Script**: `Phase1_Step1B_RepresentativeSubjects_Plots.py`

### Purpose

Auto-select one representative subject per band dominance category and generate comprehensive figures.

### Selection Criteria

For each category (theta, mu, beta_low, beta_high):
1. Filter: `best_band == category`
2. Sort by: `best_minus_baseline` (largest improvement)
3. Pick: Top subject

### Generated Plots (per subject)

**Plot 1: Time-Frequency ERD Contrast**
- TFR map (4-30 Hz, baseline %)
- Contrast: T1 - T2
- Averaged across motor-strip channels

**Plot 2: Band Discriminability Bar Chart**
- Mean |ERD_T1 - ERD_T2| per band
- Across all motor-strip channels

**Plot 3: Epoch PSD Overlay**
- Welch PSD for T1 vs T2
- Shaded band regions
- Averaged across channels

**Bonus: Montage Snapshot**
- Sensor positions for motor-strip ROI

### Output Structure

```
Phase1_Step1B_RepresentativeSubjects_Plots_figs/
├── S049_mu/
│   ├── S049_Plot1_TFR_Contrast_T1minusT2.png
│   ├── S049_Plot2_BandDiscriminability.png
│   ├── S049_Plot3_PSD_T1vsT2.png
│   └── S049_Montage_motor_strip.png
├── S016_beta_low/
│   └── ...
└── S058_beta_high/
    └── ...
```

### Usage

```python
# Run after Step 1B completes
python Phase1_Step1B_RepresentativeSubjects_Plots.py

# Output directory created automatically
# Check console for selected subjects per category
```

### Configuration

```python
# TFR settings (lines ~40-41)
tfr_freqs = np.arange(4.0, 31.0, 1.0)  # 4-30 Hz, 1 Hz steps
tfr_n_cycles = tfr_freqs / 2.0         # Time-frequency resolution

# Output DPI (lines in plot functions)
fig.savefig(fig_path, dpi=200)  # Change to 300 for publication quality
```

---

## 5. Quality Control Plots

**Script**: `Compare_S049_vs_S016_vs_S058_QCPlots.py`

### Purpose

Visual comparison between strong responders and non-responders to validate preprocessing quality.

### Generated Plots

1. **Static Raw Overlay** (C3/Cz/C4)
   - First 20 seconds of cleaned continuous EEG
   - Visual check for residual artifacts

2. **Montage Plot**
   - Sensor positions (standard_1020)
   - Verify channel locations

3. **Raw PSD Overlay**
   - Mean power across C3/Cz/C4
   - Comparison between subjects
   - Shaded mu/beta bands

4. **Evoked Responses** (per subject)
   - T1 vs T2 at C3/Cz/C4
   - Time-locked averages

5. **Epoch PSD Overlay** (per subject)
   - T1 (blue) vs T2 (red)
   - Welch method
   - Shaded band regions

### Usage

```python
# Edit subjects and run (lines ~26-27)
subjects = ["S049", "S058"]  # Strong vs non-responder
run = "R05"                  # Task 3, first run

# Run comparison
python Compare_S049_vs_S016_vs_S058_QCPlots.py

# Interactive mode (optional, lines ~31-32)
SHOW_INTERACTIVE_RAW = True
SHOW_INTERACTIVE_EPOCHS = True
```

### Typical Observations

**Strong responders** (e.g., S049):
- Clear ERD in mu/beta during T1 or T2
- Distinct PSD differences between conditions
- Low noise floor in PSD

**Non-responders** (e.g., S058):
- Minimal ERD or inconsistent patterns
- Overlapping T1/T2 PSD curves
- Higher noise or muscle contamination

---

## ERD Computation Details

### Hilbert Transform Method

Used in all analysis scripts:

```python
def erd_percent(epochs, band):
    # 1. Bandpass filter
    ep = epochs.filter(band[0], band[1], fir_design="firwin")
    
    # 2. Hilbert transform -> analytic signal
    ep.apply_hilbert(envelope=False)
    
    # 3. Power = |analytic|²
    data = ep.get_data()
    power = np.abs(data) ** 2
    
    # 4. Average in baseline and task windows
    p_base = power[..., baseline_mask].mean(axis=-1)
    p_task = power[..., task_mask].mean(axis=-1)
    
    # 5. ERD% = (P_task - P_base) / P_base × 100
    erd = (p_task - p_base) / (p_base + 1e-12) * 100.0
    
    return erd  # (n_epochs, n_channels)
```

### Interpretation

- **Negative ERD**: Desynchronization (expected for motor activity)
- **Positive ERD**: Synchronization (unexpected, may indicate artifacts)
- **Near-zero ERD**: No modulation (non-responder or wrong band)

### Best Practices

1. **Baseline window**: Should precede event (typically -0.5 to 0 s)
2. **Task window**: Capture sustained activity (0.5 to 1.5 s for execution, adjust for imagery)
3. **Band selection**: Mu (8-13 Hz) and beta (13-30 Hz) most reliable for motor tasks
4. **Channel selection**: Contralateral motor cortex strongest for unilateral movements; bilateral channels (C3, Cz, C4) for bilateral tasks

---

## Workflow Summary

### Complete Analysis Workflow

```bash
# 1. Preprocessing (once)
python EDIH_Preprocessing_v0_1.py
# → Creates cleaned-dataset/SXXX/ directories

# 2. Responder screening
python Task3_responder_screen_S001_S109.py
# → Outputs: task3_responder_screen_S001_S109.csv

# 3. Channel set comparison (Step 1A)
python Phase1_Step1A_ExtendedChannel_ERD_Discriminability_Task3.py
# → Outputs: Phase1_Step1A_Task3_ERD_discriminability_min_vs_ext.csv

# 4. Multi-band analysis (Step 1B)
python Phase1_Step1B_MultiBand_Discriminability_Task3.py
# → Outputs: Phase1_Step1B_Task3_band_discriminability_motor_strip.csv

# 5. Generate plots for representative subjects
python Phase1_Step1B_RepresentativeSubjects_Plots.py
# → Outputs: Phase1_Step1B_RepresentativeSubjects_Plots_figs/

# 6. QC comparison plots (optional)
python Compare_S049_vs_S016_vs_S058_QCPlots.py
# → Interactive plots
```

### Parallel Processing

For large-scale analysis:

```python
# Split subjects across multiple processes
# In Phase1_Step1B_MultiBand_Discriminability_Task3.py

# Process 1:
subjects = [f"S{i:03d}" for i in range(1, 28)]

# Process 2:
subjects = [f"S{i:03d}" for i in range(28, 55)]

# Process 3:
subjects = [f"S{i:03d}" for i in range(55, 82)]

# Process 4:
subjects = [f"S{i:03d}" for i in range(82, 110)]

# Combine results:
import pandas as pd
df = pd.concat([
    pd.read_csv("batch1.csv"),
    pd.read_csv("batch2.csv"),
    pd.read_csv("batch3.csv"),
    pd.read_csv("batch4.csv")
])
df.to_csv("combined_results.csv", index=False)
```

---

## Troubleshooting

### Common Issues

1. **"No usable epochs found"**
   - Check that R05/R09/R13 epochs exist
   - Verify T1/T2 events in epochs file: `epochs.event_id`

2. **"Missing channels"**
   - Normal for some subjects (not all have full 64-ch montage)
   - Pipeline adapts to available channels
   - Check if `REQUIRE_ALL_MOTOR_STRIP = True` is too strict

3. **NaN in results**
   - Insufficient epochs for condition (T1 or T2)
   - Bad data quality (excessive artifacts)
   - Check preprocessing logs

4. **Low discriminability across all bands**
   - Subject may be non-responder
   - Verify data quality with QC plots
   - Consider removing from analysis

### Recommendations

- **Top 30-40 subjects**: Focus on dual responders for BCI development
- **Band-specific subjects**: Useful for understanding neural mechanisms
- **Validation**: Always cross-validate with Step 2 (CSP+LDA decoding)

---

## Next Steps: Phase 2

After completing Phase 1, proceed to:

1. **CSP Feature Extraction**: Apply Common Spatial Patterns
2. **LDA Classification**: Train Linear Discriminant Analysis
3. **Cross-Validation**: 5-fold or leave-one-run-out
4. **Band Validation**: Confirm Step 1B findings with actual decoding accuracy

**Expected correlation**: High ERD discriminability (Phase 1) → High decoding accuracy (Phase 2)

---

## References

- **ERD/ERS**: Pfurtscheller & Lopes da Silva (1999). *Clinical Neurophysiology*
- **Motor Imagery BCI**: Blankertz et al. (2008). *NeuroImage*
- **CSP**: Ramoser et al. (2000). *IEEE Trans. Biomed. Eng.*

---

**Last updated**: February 2026
