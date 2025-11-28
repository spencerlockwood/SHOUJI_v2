#!/bin/bash

# Script to download test data
# For the actual paper data, you would need to download from EMBL-ENA
# This script creates synthetic data for testing

echo "Setting up test data directory..."

# Create subdirectories
mkdir -p synthetic
mkdir -p real

echo "Generating synthetic test data..."

# Create a Python script to generate test data
cat > generate_synthetic.py << 'EOF'
import random
import sys

def generate_sequence(length, alphabet='ACGT'):
    return ''.join(random.choice(alphabet) for _ in range(length))

def introduce_edits(seq, num_edits):
    seq_list = list(seq)
    for _ in range(num_edits):
        if len(seq_list) == 0:
            break
        pos = random.randint(0, len(seq_list)-1)
        edit_type = random.choice(['sub', 'ins', 'del'])
        
        if edit_type == 'sub':
            seq_list[pos] = random.choice('ACGT')
        elif edit_type == 'ins':
            seq_list.insert(pos, random.choice('ACGT'))
        elif edit_type == 'del' and len(seq_list) > 1:
            seq_list.pop(pos)
    
    return ''.join(seq_list)

# Generate test sets
random.seed(42)

configs = [
    ('set_100bp_e2.txt', 100, 2, 1000),
    ('set_100bp_e5.txt', 100, 5, 1000),
    ('set_150bp_e4.txt', 150, 4, 1000),
    ('set_250bp_e5.txt', 250, 5, 1000),
]

for filename, seq_len, max_edits, num_pairs in configs:
    print(f"Generating {filename}...")
    with open(f'synthetic/{filename}', 'w') as f:
        f.write(f"# Synthetic dataset: length={seq_len}, max_edits={max_edits}\n")
        f.write("# Format: text<TAB>pattern<TAB>num_edits\n")
        
        for i in range(num_pairs):
            text = generate_sequence(seq_len)
            num_edits = random.randint(0, max_edits * 2)
            pattern = introduce_edits(text, num_edits)
            
            # Pad/trim to correct length
            if len(pattern) < seq_len:
                pattern += generate_sequence(seq_len - len(pattern))
            elif len(pattern) > seq_len:
                pattern = pattern[:seq_len]
            
            f.write(f"{text}\t{pattern}\t{num_edits}\n")

print("Synthetic data generation complete!")
EOF

python3 generate_synthetic.py
rm generate_synthetic.py

echo ""
echo "Test data setup complete!"
echo "Synthetic data located in: data/synthetic/"
echo ""
echo "To download real data from the paper (ERR240727_1, etc.):"
echo "Visit: https://www.ebi.ac.uk/ena"
echo "Search for accession numbers: ERR240727, SRR826460, SRR826471"
echo ""