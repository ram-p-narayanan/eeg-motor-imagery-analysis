# EEG Motor Imagery Analysis Pipeline

A comprehensive Python pipeline for preprocessing, analyzing, and validating Event-Related Desynchronization (ERD) patterns in the PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB).

## Overview

This repository contains tools for analyzing motor imagery EEG data, focusing on identifying subject-specific optimal frequency bands for brain-computer interface (BCI) applications. The pipeline performs automated preprocessing, artifact removal, and multi-band ERD discriminability analysis across 109 subjects.

## Key Features

- **Automated Preprocessing**: Batch processing with ICA-based artifact removal
- **ERD Analysis**: Multi-band frequency analysis (theta, mu, beta_low, beta_high)
- **Subject Screening**: Automated responder classification based on ERD patterns
- **Channel Set Comparison**: Minimal vs. extended sensorimotor channel analysis
- **Quality Control**: Comprehensive visualization tools for data validation
- **Proxy ECG Extraction**: Heartbeat component reconstruction from ICA sources

## Dataset

This pipeline is designed for the **PhysioNet EEG Motor Movement/Imagery Database**:
- 109 subjects (S001-S109)
- 64-channel EEG (10-20 system)
- 14 runs per subject (R01-R14)
- Sampling rate: 160 Hz
- Tasks include: baseline (eyes open/closed), motor execution, and motor imagery

**Task 3 Focus**: Real motor execution
- T1: Both fists clenching
- T2: Both feet movement

## Installation

### Requirements

```bash
# Core dependencies
mne >= 1.5.0
numpy >= 1.24.0
pandas >= 2.0.0
matplotlib >= 3.7.0
scipy >= 1.10.0

# Optional but recommended
mne-icalabel >= 0.4.0  # For ICLabel component classification
autoreject >= 0.4.0    # For automated epoch rejection
openpyxl >= 3.1.0      # For Excel file I/O
```

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/eeg-motor-imagery-analysis.git
cd eeg-motor-imagery-analysis

# Create conda environment
conda create -n eeg_analysis python=3.10
conda activate eeg_analysis

# Install dependencies
pip install mne numpy pandas matplotlib scipy openpyxl
pip install mne-icalabel autoreject
```

## Project Structure

```
eeg-motor-imagery-analysis/
│
├── README.md                                    # This file
├── preprocess_eegmmidb.py                      # Core preprocessing utilities
├── EDIH_Preprocessing_v0_1.py                  # Main preprocessing pipeline
│
├── Phase1_Step1A_ExtendedChannel_ERD_Discriminability_Task3.py
├── Phase1_Step1B_MultiBand_Discriminability_Task3.py
├── Phase1_Step1B_RepresentativeSubjects_Plots.py
├── Task3_responder_screen_S001_S109.py
├── Compare_S049_vs_S016_vs_S058_QCPlots.py
│
├── docs/
│   ├── PREPROCESSING.md                        # Preprocessing documentation
│   ├── ANALYSIS.md                             # Analysis pipeline documentation
│   └── VISUALIZATION.md                        # Visualization guide
│
└── examples/
    └── example_workflow.md                     # End-to-end example
```

## Quick Start

### 1. Data Preparation

Download the EEGMMIDB dataset from PhysioNet and organize as follows:

```
data_root/
└── files/
    ├── S001/
    │   ├── S001R01.edf
    │   ├── S001R02.edf
    │   └── ...
    ├── S002/
    └── ...
```

### 2. Preprocessing

```python
# Edit paths in EDIH_Preprocessing_v0_1.py
data_root = Path("path/to/eeg-motor-movementimagery-dataset-1.0.0/files")
clean_root = Path("path/to/cleaned-dataset")

# Run preprocessing
python EDIH_Preprocessing_v0_1.py
```

**Output**: Cleaned raw files and epochs for each subject/run in `cleaned-dataset/SXXX/`

### 3. Responder Screening

```python
# Screen subjects for Task 3 motor imagery responses
python Task3_responder_screen_S001_S109.py
```

**Output**: `task3_responder_screen_S001_S109.csv` with dual-responder rankings

### 4. Multi-Band Analysis

```python
# Step 1A: Compare minimal vs extended channel sets
python Phase1_Step1A_ExtendedChannel_ERD_Discriminability_Task3.py

# Step 1B: Test subject-specific optimal frequency bands
python Phase1_Step1B_MultiBand_Discriminability_Task3.py

# Generate representative subject plots
python Phase1_Step1B_RepresentativeSubjects_Plots.py
```

**Outputs**:
- CSV summaries with discriminability metrics
- TFR contrast maps
- Band discriminability bar charts
- PSD overlays (T1 vs T2)

## Analysis Pipeline

### Phase 1: Pre-CSP ERD Validation

#### Step 1A: Channel Set Comparison
- **Goal**: Determine if extended sensorimotor channels improve discriminability
- **Channels**: 
  - Minimal: C3, Cz, C4
  - Extended: FC5/3/1/z/2/4/6, C5/3/1/z/2/4/6, CP5/3/1/z/2/4/6
- **Metrics**: Max and mean |ERD_T1 - ERD_T2| for mu (8-13 Hz) and beta (13-30 Hz)

#### Step 1B: Multi-Band Discriminability
- **Goal**: Identify subject-specific optimal frequency bands
- **Bands Tested**:
  - Theta: 4-8 Hz
  - Mu: 8-13 Hz
  - Beta_low: 13-20 Hz
  - Beta_high: 20-30 Hz
  - Beta: 13-30 Hz (baseline)
  - Mu_beta: 8-30 Hz (baseline)
- **ROI**: Motor strip (C5, C3, C1, Cz, C2, C4, C6)

### Preprocessing Details

1. **Loading**: Read EDF files, keep EEG channels only
2. **Channel standardization**: Remove 'EEG ', '-REF' prefixes
3. **Montage**: Apply standard_1020 montage
4. **Notch filtering**: 50 Hz (powerline noise)
5. **Bad channel detection**: Z-score and flatness-based
6. **Bandpass for ICA**: 1-40 Hz
7. **ICA**: Extended Infomax (or Infomax)
8. **Component classification**: ICLabel (eye, muscle, heart, line noise removal)
9. **Final filtering**: 0.5-40 Hz
10. **Epoching**: -0.5 to 4.0 s, baseline correction (-0.5 to 0 s)
11. **AutoReject**: Automated epoch rejection (optional)

## Key Findings

Based on preliminary analysis:

- **Extended channels**: Improved discriminability in ~60-70% of subjects
- **Subject-specific bands**: ~40% of subjects show optimal responses in narrow beta sub-bands
- **Dual responders**: Subjects with strong ERD in both mu and beta bands are ideal for BCI applications

## Visualization Examples

The pipeline generates multiple visualization types:

1. **Time-Frequency ERD Contrast Maps**: T1 vs T2 differences across time and frequency
2. **Band Discriminability Bar Charts**: Comparison across frequency bands
3. **PSD Overlays**: Power spectral density for T1 vs T2
4. **Quality Control Plots**: Raw signal, montage, and evoked response comparisons

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{eeg_motor_imagery_pipeline,
  author = {Your Name},
  title = {EEG Motor Imagery Analysis Pipeline},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/eeg-motor-imagery-analysis}
}
```

And the original dataset:

```bibtex
@article{schalk2004bci2000,
  title={BCI2000: a general-purpose brain-computer interface (BCI) system},
  author={Schalk, Gerwin and McFarland, Dennis J and Hinterberger, Thilo and Birbaumer, Niels and Wolpaw, Jonathan R},
  journal={IEEE Transactions on biomedical engineering},
  volume={51},
  number={6},
  pages={1034--1043},
  year={2004},
  publisher={IEEE}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - See LICENSE file for details


## Acknowledgments

- PhysioNet for providing the EEGMMIDB dataset
- MNE-Python development team
- ICLabel and AutoReject developers

---

**Note**: This is a research tool. Results should be validated before clinical or commercial use.
