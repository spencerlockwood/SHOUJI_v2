# Shouji Pre-alignment Filter Reimplementation

Reimplementation of the Shouji pre-alignment filter from:
Alser et al. (2019) "Shouji: a fast and efficient pre-alignment filter for sequence alignment"

## Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download test data
cd data
bash download_data.sh
cd ..
```

## Running Experiments
```bash
# Run all experiments and generate plots
python analysis/run_experiments.py

# Generate comparison plots
python analysis/generate_plots.py
```

## Project Structure

- `shouji/` - Core Shouji implementation
- `baseline/` - Baseline alignment tools (Edlib)
- `data/` - Test datasets
- `tests/` - Unit tests and validation
- `analysis/` - Experiment scripts and plotting