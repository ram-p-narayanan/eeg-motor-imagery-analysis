# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of EEG Motor Imagery Analysis Pipeline
- Preprocessing utilities with ICA-based artifact removal
- Multi-band ERD discriminability analysis
- Responder screening for dual-band motor responses
- Channel set comparison (minimal vs extended)
- Representative subject visualization tools
- Quality control comparison plots
- Comprehensive documentation (README, guides, examples)

## [1.0.0] - 2025-02-05

### Added
- Core preprocessing pipeline (`preprocess_eegmmidb.py`)
- Batch preprocessing script (`EDIH_Preprocessing_v0_1.py`)
- Responder screening (`Task3_responder_screen_S001_S109.py`)
- Channel comparison analysis (`Phase1_Step1A_ExtendedChannel_ERD_Discriminability_Task3.py`)
- Multi-band analysis (`Phase1_Step1B_MultiBand_Discriminability_Task3.py`)
- Automated plotting (`Phase1_Step1B_RepresentativeSubjects_Plots.py`)
- QC comparison tool (`Compare_S049_vs_S016_vs_S058_QCPlots.py`)

### Features
- Extended Infomax ICA for artifact removal
- ICLabel-based component classification
- Proxy ECG extraction from ICA sources
- AutoReject integration for epoch cleaning
- Bad channel detection and interpolation
- Hilbert-based ERD% computation
- Time-frequency analysis with Morlet wavelets
- Multi-band discriminability metrics

### Documentation
- README.md with project overview
- PREPROCESSING.md with detailed preprocessing guide
- ANALYSIS.md with complete analysis pipeline documentation
- QUICKSTART.md for 5-minute setup
- EXAMPLE_WORKFLOW.md with end-to-end example
- CONTRIBUTING.md with contribution guidelines

### Supported Datasets
- PhysioNet EEG Motor Movement/Imagery Database (EEGMMIDB)
- 109 subjects, 64-channel EEG, 14 runs per subject

---

## Future Releases

### [1.1.0] - Planned
- [ ] Command-line interface with argparse
- [ ] Progress bars for long-running operations
- [ ] Parallel processing support
- [ ] Docker container for reproducibility

### [2.0.0] - Planned (Phase 2)
- [ ] CSP (Common Spatial Patterns) feature extraction
- [ ] LDA (Linear Discriminant Analysis) classification
- [ ] Cross-validation framework
- [ ] Performance metrics and validation
- [ ] Real-time BCI demo

### [3.0.0] - Future
- [ ] Deep learning models (EEGNet, ShallowConvNet)
- [ ] Riemannian geometry classifiers
- [ ] Multi-dataset support (BCI Competition, BNCI Horizon)
- [ ] Interactive web dashboard

---

## Notes

### Version Numbering
- **Major version (X.0.0)**: Breaking changes, incompatible API updates
- **Minor version (0.X.0)**: New features, backward compatible
- **Patch version (0.0.X)**: Bug fixes, documentation updates

### Contribution
See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to this project.
