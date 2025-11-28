# Quick Start Guide

## Installation
```bash
# Clone repository
git clone <your-repo-url>
cd shouji-reimplementation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup data
cd data
bash download_data.sh
cd ..
```

## Run Basic Tests
```bash
# Run unit tests
python -m pytest tests/test_shouji.py -v

# Validate accuracy
python tests/validate_accuracy.py
```

## Run Full Experiments
```bash
# Run all experiments (may take 30-60 minutes)
python analysis/run_experiments.py

# Generate plots
python analysis/generate_plots.py

# Compare with paper
python analysis/compare_results.py
```

## View Results

Results will be saved in:
- `results/` - JSON files with experimental data
- `plots/` - PNG images of all figures

## Expected Outputs

1. **False Accept Rate plots** - Should show low FAR (<15% for most configs)
2. **False Reject Rate plots** - Should be 0% (key feature of Shouji)
3. **Execution time comparison** - Shouji should be faster than Edlib
4. **Confusion matrices** - Visual accuracy representation

## Troubleshooting

If you encounter memory errors with large datasets:
- Reduce `num_pairs` in `run_experiments.py`
- Test with synthetic data first before real data

If plots don't generate:
- Ensure matplotlib and seaborn are installed
- Check that experiments completed successfully