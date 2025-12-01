#!/usr/bin/env python3
"""
Generate sequence pairs from real FASTQ reads
Simulates what mrFAST would generate before alignment
"""

import random
import sys
from typing import List, Tuple
from Bio import SeqIO
import gzip

def read_fastq(filename: str, max_reads: int = 100000) -> List[str]:
    """Read sequences from FASTQ file"""
    sequences = []
    
    if filename.endswith('.gz'):
        handle = gzip.open(filename, 'rt')
    else:
        handle = open(filename, 'r')
    
    print(f"Reading {filename}...")
    for i, record in enumerate(SeqIO.parse(handle, "fastq")):
        if i >= max_reads:
            break
        sequences.append(str(record.seq))
        
        if (i + 1) % 10000 == 0:
            print(f"  Read {i + 1} sequences...")
    
    handle.close()
    print(f"  Total: {len(sequences)} sequences")
    return sequences

def introduce_edits(seq: str, num_edits: int) -> str:
    """Introduce random edits to create dissimilar pairs"""
    if num_edits == 0:
        return seq
    
    seq_list = list(seq)
    alphabet = 'ACGT'
    
    for _ in range(min(num_edits, len(seq_list))):
        edit_type = random.choice(['substitute', 'insert', 'delete'])
        if not seq_list:
            break
            
        pos = random.randint(0, len(seq_list) - 1)
        
        if edit_type == 'substitute':
            original = seq_list[pos]
            seq_list[pos] = random.choice([c for c in alphabet if c != original])
        elif edit_type == 'insert':
            seq_list.insert(pos, random.choice(alphabet))
        elif edit_type == 'delete' and len(seq_list) > 1:
            seq_list.pop(pos)
    
    return ''.join(seq_list)

def generate_pairs(sequences: List[str], num_pairs: int, 
                   max_edits: int, seq_length: int) -> List[Tuple[str, str, int]]:
    """Generate sequence pairs with varying edit distances"""
    pairs = []
    
    print(f"Generating {num_pairs} pairs (max_edits={max_edits})...")
    
    for i in range(num_pairs):
        # Pick a random read as reference
        ref = random.choice(sequences)
        
        # Ensure correct length
        if len(ref) < seq_length:
            continue
        ref = ref[:seq_length]
        
        # Randomly decide number of edits (0 to 2*max_edits for mix)
        num_edits = random.randint(0, max_edits * 2)
        
        # Create query with edits
        query = introduce_edits(ref, num_edits)
        
        # Pad/trim to correct length
        if len(query) < seq_length:
            query += ''.join(random.choice('ACGT') for _ in range(seq_length - len(query)))
        elif len(query) > seq_length:
            query = query[:seq_length]
        
        pairs.append((ref, query, num_edits))
        
        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1} pairs...")
    
    return pairs

def main():
    random.seed(42)
    
    # Configuration matching paper
    configs = [
        {
            'input': 'real_data/ERR240727_1.fastq.gz',
            'seq_length': 100,
            'datasets': [
                ('set_1_100bp_e2.txt', 2, 30000),
                ('set_2_100bp_e5.txt', 5, 30000),
            ]
        },
        {
            'input': 'real_data/SRR826460_1.fastq.gz',
            'seq_length': 150,
            'datasets': [
                ('set_5_150bp_e4.txt', 4, 30000),
                ('set_6_150bp_e7.txt', 7, 30000),
            ]
        },
        {
            'input': 'real_data/SRR826471_1.fastq.gz',
            'seq_length': 250,
            'datasets': [
                ('set_9_250bp_e8.txt', 8, 30000),
                ('set_10_250bp_e15.txt', 15, 30000),
            ]
        },
    ]
    
    for config in configs:
        input_file = config['input']
        seq_length = config['seq_length']
        
        # Read sequences from FASTQ
        try:
            sequences = read_fastq(input_file, max_reads=100000)
        except FileNotFoundError:
            print(f"Warning: {input_file} not found, skipping...")
            continue
        
        # Generate datasets
        for output_file, max_edits, num_pairs in config['datasets']:
            print(f"\nGenerating {output_file}...")
            pairs = generate_pairs(sequences, num_pairs, max_edits, seq_length)
            
            # Write to file
            output_path = f'real_data/{output_file}'
            with open(output_path, 'w') as f:
                f.write(f"# Real data from {input_file}\n")
                f.write(f"# seq_length={seq_length}, max_edits={max_edits}\n")
                f.write("# Format: reference<TAB>query<TAB>introduced_edits\n")
                
                for ref, query, edits in pairs:
                    f.write(f"{ref}\t{query}\t{edits}\n")
            
            print(f"  Saved to {output_path}")
    
    print("\n✓ All datasets generated!")

if __name__ == '__main__':
    main()
