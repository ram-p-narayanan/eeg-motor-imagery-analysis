# Example Workflow: Complete Analysis

This document walks through a complete analysis from raw data to publication-ready results.

## Scenario

You want to develop a motor imagery BCI for stroke rehabilitation. Your goals:

1. Identify subjects with strong, reliable ERD patterns
2. Determine optimal frequency bands per subject
3. Select best channel montage
4. Generate figures for publication

**Dataset**: EEGMMIDB (already downloaded)

---

## Step 1: Preprocessing (Day 1)

### Configure Paths

```python
# EDIH_Preprocessing_v0_1.py (lines ~176-180)
data_root = Path("/data/eegmmidb/files")
clean_root = Path("/data/cleaned-dataset")
```

### Test on Small Batch

```python
# Line ~182: Test with 3 subjects first
subject_ids = range(1, 4)  # S001-S003
```

Run:
```bash
python EDIH_Preprocessing_v0_1.py
```

**Output check**:
```bash
ls cleaned-dataset/S001/
# Expected:
# S001_bad_channels.txt
# S001R01-cleaned_raw.fif ... S001R14-cleaned_raw.fif
# S001R03-epo.fif ... S001R14-epo.fif
# S001R03_proxy_ecg_raw.fif ... (if ICLabel detected heartbeat)
```

**Review logs**:
```bash
cat cleaned-dataset/S001/S001_bad_channels.txt
```

Expected: 0-3 bad channels per run. If >5, check data quality.

### Scale Up to All Subjects

```python
# Line ~182: Process all subjects
subject_ids = range(1, 110)  # S001-S109
```

Run overnight:
```bash
nohup python EDIH_Preprocessing_v0_1.py > preprocessing.log 2>&1 &
```

**Time estimate**: 6-8 hours (single-threaded)

---

## Step 2: Quality Control (Day 2 Morning)

### Visual Inspection

```bash
python Compare_S049_vs_S016_vs_S058_QCPlots.py
```

**Check**:
1. Raw overlay: Should be clean, no huge spikes
2. PSD: Should show alpha peak (~10 Hz), no line noise
3. Evoked: T1 vs T2 should show visible differences (good subjects)

**Red flags**:
- Raw signal flatlines or saturates → Electrode issue
- PSD dominated by 50 Hz → Notch filter failed
- Evoked T1 ≈ T2 → Subject may be non-responder

If QC fails, check:
```bash
# Review bad channels for problematic subjects
grep "S049" cleaned-dataset/*/S*_bad_channels.txt

# Re-run preprocessing for specific subject if needed
# (comment out subject_ids loop, hard-code subject)
```

---

## Step 3: Responder Screening (Day 2 Afternoon)

### Update Paths

All analysis scripts need:
```python
# Line ~38-42 in each script
clean_root = Path("/data/cleaned-dataset")
```

### Run Screening

```bash
python Task3_responder_screen_S001_S109.py
```

**Time**: ~15-30 minutes

### Analyze Results

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("task3_responder_screen_S001_S109.csv")

# How many dual responders?
dual = df[df["dual_responder"] == True]
print(f"Dual responders: {len(dual)}/{len(df)} ({100*len(dual)/len(df):.1f}%)")

# Distribution of scores
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(df["mu_best_score"], bins=20, alpha=0.7, label="Mu")
plt.xlabel("Mu score")
plt.ylabel("Count")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(df["beta_best_score"], bins=20, alpha=0.7, label="Beta", color="orange")
plt.xlabel("Beta score")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("responder_score_distribution.png", dpi=150)
plt.show()

# Top 10 for BCI development
top10 = dual.sort_values("dual_total_score", ascending=False).head(10)
print("\nTop 10 subjects for BCI:")
print(top10[["subject", "dual_total_score", "mu_best_channel", "beta_best_channel"]])
```

**Expected**: 40-60% dual responders for healthy adults

---

## Step 4: Channel Set Comparison (Day 3 Morning)

```bash
python Phase1_Step1A_ExtendedChannel_ERD_Discriminability_Task3.py
```

**Time**: ~1-2 hours

### Analyze Results

```python
df = pd.read_csv("Phase1_Step1A_Task3_ERD_discriminability_min_vs_ext.csv")

# Did extended channels help overall?
valid = df.dropna(subset=["delta_combo"])
improved = valid[valid["delta_combo"] > 0]
print(f"Extended improved: {len(improved)}/{len(valid)} subjects ({100*len(improved)/len(valid):.1f}%)")

# Magnitude of improvement
import numpy as np
print(f"Mean improvement: {valid['delta_combo'].mean():.3f}")
print(f"Median improvement: {valid['delta_combo'].median():.3f}")

# Visualize
plt.figure(figsize=(8, 5))
plt.scatter(valid["min_combo_score"], valid["ext_combo_score"], alpha=0.6)
plt.plot([0, 50], [0, 50], 'r--', label="No change")
plt.xlabel("Minimal set discriminability")
plt.ylabel("Extended set discriminability")
plt.title("Minimal vs Extended Channel Sets")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("minimal_vs_extended_scatter.png", dpi=150)
plt.show()

# Subject-specific decision
for _, row in df.head(10).iterrows():
    use_ext = "✓" if row["delta_combo"] > 2.0 else "✗"
    print(f"{row['subject']}: delta={row['delta_combo']:5.2f} → Use extended: {use_ext}")
```

**Decision rule**: Use extended channels if `delta_combo > 2.0` (at least 2% improvement)

---

## Step 5: Multi-Band Analysis (Day 3 Afternoon)

```bash
python Phase1_Step1B_MultiBand_Discriminability_Task3.py
```

**Time**: ~2-3 hours

### Analyze Results

```python
df = pd.read_csv("Phase1_Step1B_Task3_band_discriminability_motor_strip.csv")

# Band dominance
print("\nBand dominance distribution:")
print(df["best_band"].value_counts())

# Improvement over baseline
print(f"\nMean improvement over mu_beta: {df['best_minus_baseline'].mean():.3f}")
print(f"Subjects with >5% improvement: {len(df[df['best_minus_baseline'] > 5])}")

# Visualize band preferences
import seaborn as sns
plt.figure(figsize=(10, 6))
order = ["theta", "mu", "beta_low", "beta_high", "beta", "mu_beta"]
counts = df["best_band"].value_counts().reindex(order, fill_value=0)
sns.barplot(x=counts.index, y=counts.values, palette="viridis")
plt.xlabel("Frequency Band")
plt.ylabel("Number of Subjects")
plt.title("Subject-Specific Optimal Bands (N={})".format(len(df)))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("band_dominance_distribution.png", dpi=150)
plt.show()

# Per-subject band recommendations
print("\nTop 20 subjects with band recommendations:")
top20 = df.sort_values("best_minus_baseline", ascending=False).head(20)
for _, row in top20.iterrows():
    print(f"{row['subject']}: {row['best_band']:10s} "
          f"(improvement: {row['best_minus_baseline']:+5.2f}%, "
          f"discriminability: {row['best_maxdiff']:5.1f}%)")
```

---

## Step 6: Generate Figures (Day 4)

```bash
python Phase1_Step1B_RepresentativeSubjects_Plots.py
```

**Time**: ~30-60 minutes

**Output**: Folder `Phase1_Step1B_RepresentativeSubjects_Plots_figs/` with subfolders per subject

### Review Plots

Navigate to output directory:
```bash
cd Phase1_Step1B_RepresentativeSubjects_Plots_figs/
ls
# Expected: S049_mu/ S016_beta_low/ S058_beta_high/ S023_theta/
```

**For each subject**, check:

1. **Plot 1 (TFR contrast)**: 
   - Strong blue (negative) = ERD
   - Should be concentrated in optimal band during task (0.5-1.5 s)

2. **Plot 2 (Bar chart)**:
   - Tallest bar = optimal band
   - Should match `best_band` from CSV

3. **Plot 3 (PSD)**:
   - T1 (blue) and T2 (red) should diverge in optimal band
   - Shaded regions help identify band boundaries

### Publication-Quality Adjustments

Edit `Phase1_Step1B_RepresentativeSubjects_Plots.py`:

```python
# Line ~40-41: Increase frequency resolution
tfr_freqs = np.arange(4.0, 31.0, 0.5)  # 0.5 Hz steps instead of 1.0

# Lines in plot functions: Increase DPI
fig.savefig(fig_path, dpi=300)  # Instead of 200
```

Re-run for publication-ready figures.

---

## Step 7: Compile Report (Day 4 Afternoon)

### Summary Statistics

Create a summary document:

```python
import pandas as pd

# Load all results
responder = pd.read_csv("task3_responder_screen_S001_S109.csv")
step1a = pd.read_csv("Phase1_Step1A_Task3_ERD_discriminability_min_vs_ext.csv")
step1b = pd.read_csv("Phase1_Step1B_Task3_band_discriminability_motor_strip.csv")

# Create summary
summary = {
    "Total subjects processed": len(responder),
    "Dual responders (mu+beta)": len(responder[responder["dual_responder"] == True]),
    "Dual responder rate (%)": 100 * len(responder[responder["dual_responder"] == True]) / len(responder),
    "Subjects benefiting from extended channels": len(step1a[step1a["delta_combo"] > 0]),
    "Extended channel improvement rate (%)": 100 * len(step1a[step1a["delta_combo"] > 0]) / len(step1a),
    "Mean band-specific improvement (%)": step1b["best_minus_baseline"].mean(),
    "Subjects with narrow-beta dominance": len(step1b[step1b["best_band"].isin(["beta_low", "beta_high"])]),
}

# Print summary
print("\n=== Analysis Summary ===")
for key, val in summary.items():
    if isinstance(val, float):
        print(f"{key}: {val:.2f}")
    else:
        print(f"{key}: {val}")

# Export for paper
with open("analysis_summary.txt", "w") as f:
    for key, val in summary.items():
        f.write(f"{key}: {val}\n")
```

### Tables for Paper

**Table 1: Top 10 Subjects for BCI Development**

```python
dual = responder[responder["dual_responder"] == True]
top10 = dual.sort_values("dual_total_score", ascending=False).head(10)

# Merge with band info
top10 = top10.merge(step1b[["subject", "best_band", "best_maxdiff"]], on="subject")

# Format for paper
paper_table = top10[[
    "subject",
    "dual_total_score",
    "mu_best_score",
    "beta_best_score",
    "best_band",
    "best_maxdiff"
]].copy()
paper_table.columns = [
    "Subject",
    "Total Score",
    "Mu Score",
    "Beta Score",
    "Optimal Band",
    "Discriminability (%)"
]
paper_table.to_csv("table1_top_subjects.csv", index=False, float_format="%.2f")
print(paper_table.to_latex(index=False, float_format="%.2f"))
```

**Table 2: Band Dominance Summary**

```python
band_summary = step1b["best_band"].value_counts().reset_index()
band_summary.columns = ["Band", "Count"]
band_summary["Percentage"] = 100 * band_summary["Count"] / len(step1b)
band_summary = band_summary.sort_values("Count", ascending=False)
band_summary.to_csv("table2_band_distribution.csv", index=False, float_format="%.1f")
print(band_summary.to_latex(index=False, float_format="%.1f"))
```

---

## Step 8: Select Subjects for BCI Development

### Criteria

Based on analysis, select subjects for next phase (CSP+LDA classification):

```python
# Merge all results
final = responder.merge(step1a[["subject", "delta_combo"]], on="subject", how="left")
final = final.merge(step1b[["subject", "best_band", "best_minus_baseline"]], on="subject", how="left")

# Filter criteria:
# 1. Dual responder
# 2. Extended channels helpful (delta_combo > 0) OR minimal is sufficient
# 3. Subject-specific band shows improvement (best_minus_baseline > 3%)
selected = final[
    (final["dual_responder"] == True) &
    (final["best_minus_baseline"] > 3.0)
].copy()

print(f"\nSelected {len(selected)} subjects for Phase 2 (CSP+LDA):")
print(selected[["subject", "dual_total_score", "best_band", "best_minus_baseline"]].to_string(index=False))

# Export subject list
selected["subject"].to_csv("selected_subjects_phase2.txt", index=False, header=False)
```

### Subject-Specific Configurations

Create configuration file for Phase 2:

```python
import json

configs = {}
for _, row in selected.iterrows():
    configs[row["subject"]] = {
        "band": row["best_band"],
        "use_extended_channels": bool(row["delta_combo"] > 2.0),
        "expected_discriminability": float(row["best_minus_baseline"])
    }

with open("subject_configs.json", "w") as f:
    json.dump(configs, f, indent=2)

print("Saved subject-specific configurations to subject_configs.json")
```

---

## Step 9: Archive and Document

### Create Archive

```bash
mkdir analysis_output
mv *.csv analysis_output/
mv *.png analysis_output/
mv *.txt analysis_output/
mv Phase1_Step1B_RepresentativeSubjects_Plots_figs/ analysis_output/
tar -czf analysis_results_2025-02-05.tar.gz analysis_output/
```

### Git Commit

```bash
git add analysis_output/
git commit -m "Complete Phase 1 analysis: ERD validation and band selection

- Processed 109 subjects (EEGMMIDB)
- Identified 47 dual responders
- Extended channels improved 62% of subjects
- Subject-specific optimal bands determined
- Selected 35 subjects for Phase 2

Closes #1"
git push origin main
```

---

## Timeline Summary

| Day | Task | Time | Output |
|-----|------|------|--------|
| 1 | Preprocessing | 8h (overnight) | Cleaned epochs |
| 2 AM | Quality control | 2h | QC plots |
| 2 PM | Responder screening | 1h | CSV + rankings |
| 3 AM | Channel comparison | 2h | Minimal vs extended |
| 3 PM | Multi-band analysis | 3h | Band preferences |
| 4 AM | Generate figures | 1h | Publication plots |
| 4 PM | Compile report | 2h | Tables + summary |

**Total**: ~19 hours over 4 days

---

## Next Steps: Phase 2

With selected subjects and configurations, proceed to:

1. **CSP feature extraction**: Apply Common Spatial Patterns
2. **LDA classification**: Train discriminant analysis
3. **Cross-validation**: 5-fold or leave-one-run-out
4. **Performance comparison**: Validate band preferences with accuracy

---

## Troubleshooting This Workflow

### Issue: Low dual responder rate (<30%)

**Possible causes**:
- Stricter thresholds (TH_MEAN, TH_NEGPCT)
- Poor data quality (check preprocessing logs)
- Wrong task (verify R05/R09/R13 are motor execution, not imagery)

**Solutions**:
- Relax thresholds: TH_MEAN = -8.0, TH_NEGPCT = 50.0
- Re-preprocess problematic subjects
- Verify task labels in annotations

### Issue: No improvement with extended channels

**Possible causes**:
- Subjects have localized ERD (minimal channels sufficient)
- Extended channels include too much noise
- Dataset-specific (EEGMMIDB has strong central responses)

**Solutions**:
- This is actually okay! Use minimal for these subjects
- Focus on subjects where extended helps (those with delta_combo > 2)

### Issue: All subjects show mu dominance

**Possible causes**:
- Analysis windows optimized for mu (8-13 Hz)
- Beta responses weaker in motor execution (vs imagery)
- Sample bias (healthy young adults)

**Solutions**:
- This is expected for real motor execution tasks
- Motor imagery may show more beta dominance
- Validate with literature (mu typically stronger for execution)

---

**Congratulations!** You've completed a full ERD analysis pipeline. You now have:

✅ Cleaned, preprocessed EEG data  
✅ Validated responders identified  
✅ Subject-specific band preferences  
✅ Channel set recommendations  
✅ Publication-ready figures  
✅ Selected subjects for BCI development  

Ready for Phase 2: Classification! 🚀
