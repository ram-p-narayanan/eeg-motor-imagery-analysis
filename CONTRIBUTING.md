# Contributing Guidelines

Thank you for your interest in contributing to the EEG Motor Imagery Analysis Pipeline!

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

1. **Clear title**: Describe the problem in one sentence
2. **Steps to reproduce**: Minimal code to recreate the issue
3. **Expected vs actual behavior**: What should happen vs what does happen
4. **Environment details**:
   ```
   - Python version
   - MNE version
   - Operating system
   - Dataset (if using different from EEGMMIDB)
   ```
5. **Error messages**: Full traceback if applicable

**Example**:
```markdown
**Title**: ICA fails on subject S042 with "no components" error

**Steps**:
1. Run preprocessing on S042R05.edf
2. Error occurs during ICA fitting

**Expected**: ICA should fit ~60 components
**Actual**: RuntimeError: "no components returned"

**Environment**: Python 3.10, MNE 1.5.1, Windows 11

**Traceback**:
```
[paste full error here]
```
```

### Suggesting Enhancements

Open an issue with:

1. **Feature description**: What you want to add/change
2. **Use case**: Why this would be useful
3. **Proposed implementation**: How it could work (optional)

**Example**:
```markdown
**Feature**: Add support for multi-class motor imagery (left/right hand, feet, tongue)

**Use case**: Many BCI paradigms use 4-class tasks

**Implementation**: 
- Extend event_id handling in epoching
- Add multi-class discriminability metrics
- Update visualization for 3+ conditions
```

## Development Setup

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/yourusername/eeg-motor-imagery-analysis.git
cd eeg-motor-imagery-analysis

# Add upstream remote
git remote add upstream https://github.com/originalauthor/eeg-motor-imagery-analysis.git
```

### Create Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### Install Development Dependencies

```bash
conda create -n eeg_dev python=3.10
conda activate eeg_dev

pip install -e .  # Editable install
pip install pytest pytest-cov black flake8 mypy
```

## Code Standards

### Style Guide

We follow **PEP 8** with some modifications:

```python
# Line length: 100 characters (not 79)
# Use Black formatter with default settings

# Type hints encouraged
def compute_erd(epochs: mne.Epochs, band: tuple[float, float]) -> np.ndarray:
    ...

# Docstrings: NumPy style
def example_function(param1, param2):
    """
    Short description.

    Parameters
    ----------
    param1 : type
        Description.
    param2 : type
        Description.

    Returns
    -------
    type
        Description.
    """
    ...
```

### Format Code

```bash
# Auto-format with Black
black *.py

# Check style
flake8 *.py --max-line-length=100

# Type checking (optional)
mypy *.py
```

### Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=./ tests/
```

## Pull Request Process

### 1. Make Changes

```bash
# Make your changes
# Add tests if applicable
# Update documentation

# Commit with clear messages
git add .
git commit -m "Add feature: multi-class support for motor imagery tasks"
```

### 2. Push to Fork

```bash
git push origin feature/your-feature-name
```

### 3. Open Pull Request

On GitHub, click "New Pull Request" and fill out:

**Title**: Clear, concise description

**Description**:
```markdown
## What does this PR do?
Adds support for 4-class motor imagery paradigms (left/right hand, feet, tongue).

## Why is this needed?
Many BCI studies use multi-class tasks, not just binary fists vs feet.

## Changes made:
- [ ] Extended event_id handling in epoching
- [ ] Added multi-class discriminability metrics
- [ ] Updated visualization for 3+ conditions
- [ ] Added tests for 4-class scenario
- [ ] Updated documentation

## Testing:
Tested on BCI Competition IV dataset 2a (4-class MI)

## Related issues:
Closes #42
```

### 4. Code Review

- Respond to feedback promptly
- Make requested changes
- Push updates to the same branch

### 5. Merge

Once approved, your PR will be merged!

## Contribution Ideas

### Good First Issues

- [ ] Add command-line argument parsing (argparse) to scripts
- [ ] Create Docker container for reproducible environment
- [ ] Add progress bars (tqdm) to long-running loops
- [ ] Write unit tests for preprocessing functions
- [ ] Improve error messages (more informative)

### Medium Difficulty

- [ ] Add support for other EEG datasets (BCI Competition, BNCI Horizon)
- [ ] Implement cross-validation for responder screening
- [ ] Add statistical tests (t-tests, ANOVA) to analysis
- [ ] Create interactive web dashboard (Plotly/Dash)
- [ ] Parallel processing with joblib or multiprocessing

### Advanced

- [ ] Phase 2: CSP+LDA classification pipeline
- [ ] Real-time BCI demo (online ERD classification)
- [ ] Deep learning alternatives (EEGNet, ShallowConvNet)
- [ ] Riemannian geometry classifiers
- [ ] Adaptive frequency band selection

## Documentation

### What to Document

- New functions: Add NumPy-style docstrings
- New scripts: Add header comments explaining purpose
- Configuration changes: Update README or relevant .md file
- Breaking changes: Highlight in CHANGELOG.md

### README Updates

If your change affects usage:

1. Update README.md with new instructions
2. Update relevant documentation (PREPROCESSING.md, ANALYSIS.md, QUICKSTART.md)
3. Add examples to examples/ directory

## Community Guidelines

### Be Respectful

- Assume good intentions
- Provide constructive feedback
- Welcome newcomers
- Respect diverse perspectives

### Be Clear

- Write clear code and comments
- Explain your reasoning in PRs
- Ask questions if unsure

### Be Patient

- Reviews may take time
- Maintainers are volunteers
- Follow up politely if no response after 1 week

## Questions?

- **General questions**: Open a Discussion on GitHub
- **Bug reports**: Open an Issue
- **Feature requests**: Open an Issue with "enhancement" label
- **Security issues**: Email [security@example.com] (do not open public issue)

## Recognition

Contributors will be:
- Listed in README.md (Contributors section)
- Credited in CHANGELOG.md for significant contributions
- Acknowledged in papers/presentations using this tool

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for making this project better! 🎉
