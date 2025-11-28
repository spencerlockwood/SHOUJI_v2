"""
Utility functions for sequence processing
"""

import random
from typing import List, Tuple
import numpy as np


def load_sequences(filepath: str) -> List[Tuple[str, str]]:
    """
    Load sequence pairs from file.
    
    Args:
        filepath: Path to file with sequence pairs (one pair per line, tab-separated)
        
    Returns:
        List of (text, pattern) tuples
    """
    pairs = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def generate_random_sequence(length: int, alphabet: str = 'ACGT') -> str:
    """
    Generate random DNA sequence.
    
    Args:
        length: Sequence length
        alphabet: Characters to use
        
    Returns:
        Random sequence string
    """
    return ''.join(random.choice(alphabet) for _ in range(length))


def introduce_edits(sequence: str, num_edits: int, seed: int = None) -> str:
    """
    Introduce random edits (substitutions, insertions, deletions) into sequence.
    
    Args:
        sequence: Original sequence
        num_edits: Number of edits to introduce
        seed: Random seed for reproducibility
        
    Returns:
        Modified sequence
    """
    if seed is not None:
        random.seed(seed)
    
    seq_list = list(sequence)
    alphabet = 'ACGT'
    
    for _ in range(num_edits):
        edit_type = random.choice(['substitute', 'insert', 'delete'])
        pos = random.randint(0, len(seq_list) - 1)
        
        if edit_type == 'substitute' and seq_list:
            original = seq_list[pos]
            new_char = random.choice([c for c in alphabet if c != original])
            seq_list[pos] = new_char
            
        elif edit_type == 'insert':
            seq_list.insert(pos, random.choice(alphabet))
            
        elif edit_type == 'delete' and len(seq_list) > 1:
            seq_list.pop(pos)
    
    return ''.join(seq_list)


def calculate_edit_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two sequences.
    
    Args:
        s1: First sequence
        s2: Second sequence
        
    Returns:
        Edit distance
    """
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def generate_test_dataset(num_pairs: int, seq_length: int, 
                         edit_threshold: int, seed: int = 42) -> List[Tuple[str, str, int]]:
    """
    Generate synthetic test dataset with known edit distances.
    
    Args:
        num_pairs: Number of sequence pairs to generate
        seq_length: Length of sequences
        edit_threshold: Maximum edit distance
        seed: Random seed
        
    Returns:
        List of (text, pattern, true_edit_distance) tuples
    """
    random.seed(seed)
    np.random.seed(seed)
    
    dataset = []
    
    for i in range(num_pairs):
        # Generate reference sequence
        text = generate_random_sequence(seq_length)
        
        # Randomly decide number of edits (0 to 2*edit_threshold)
        num_edits = random.randint(0, 2 * edit_threshold)
        
        # Create pattern with edits
        pattern = introduce_edits(text, num_edits, seed=seed + i)
        
        # Ensure same length for Shouji (pad/trim if needed)
        if len(pattern) < seq_length:
            pattern += generate_random_sequence(seq_length - len(pattern))
        elif len(pattern) > seq_length:
            pattern = pattern[:seq_length]
        
        # Calculate true edit distance
        true_dist = calculate_edit_distance(text, pattern)
        
        dataset.append((text, pattern, true_dist))
    
    return dataset