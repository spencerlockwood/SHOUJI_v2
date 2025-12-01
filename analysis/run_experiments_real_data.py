"""
Run experiments using real sequencing data
"""

import os
import json
import time
from typing import List, Tuple
from tqdm import tqdm

from shouji import ShoujiFilter
from baseline import EdlibBaseline


def load_sequence_pairs(filepath: str) -> List[Tuple[str, str, int]]:
    """Load sequence pairs from file"""
    pairs = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                ref, query, edits = parts[0], parts[1], int(parts[2])
                pairs.append((ref, query, edits))
    return pairs


def run_real_data_experiments():
    """Run experiments on real data"""
    
    # Real data files
    datasets = [
        ('data/real_data/set_1_100bp_e2.txt', 100, 2),
        ('data/real_data/set_2_100bp_e5.txt', 100, 5),
        ('data/real_data/set_5_150bp_e4.txt', 150, 4),
        ('data/real_data/set_6_150bp_e7.txt', 150, 7),
        ('data/real_data/set_9_250bp_e8.txt', 250, 8),
        ('data/real_data/set_10_250bp_e15.txt', 250, 15),
    ]
    
    results = {}
    
    for dataset_file, seq_length, edit_threshold in datasets:
        if not os.path.exists(dataset_file):
            print(f"Skipping {dataset_file} (not found)")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {dataset_file}")
        print(f"seq_length={seq_length}, E={edit_threshold}")
        print(f"{'='*70}")
        
        # Load data
        print("Loading sequence pairs...")
        pairs = load_sequence_pairs(dataset_file)
        print(f"Loaded {len(pairs)} pairs")
        
        # Initialize filters
        shouji = ShoujiFilter(edit_threshold, seq_length)
        edlib = EdlibBaseline(edit_threshold)
        
        # Run experiments
        tp, tn, fp, fn = 0, 0, 0, 0
        shouji_times = []
        edlib_times = []
        
        print("Processing pairs...")
        for ref, query, _ in tqdm(pairs):
            # Edlib (ground truth)
            start = time.time()
            edlib_similar, _ = edlib.align(ref, query)
            edlib_times.append(time.time() - start)
            
            # Shouji
            start = time.time()
            shouji_similar, _ = shouji.filter(ref, query)
            shouji_times.append(time.time() - start)
            
            # Count results
            if edlib_similar:
                if shouji_similar:
                    tp += 1
                else:
                    fn += 1
            else:
                if shouji_similar:
                    fp += 1
                else:
                    tn += 1
        
        # Calculate metrics
        similar_count = tp + fn
        dissimilar_count = fp + tn
        
        far = (fp / dissimilar_count * 100) if dissimilar_count > 0 else 0
        frr = (fn / similar_count * 100) if similar_count > 0 else 0
        
        import numpy as np
        avg_shouji_time = np.mean(shouji_times) * 1000
        avg_edlib_time = np.mean(edlib_times) * 1000
        
        key = f"L{seq_length}_E{edit_threshold}"
        results[key] = {
            'seq_length': seq_length,
            'edit_threshold': edit_threshold,
            'num_pairs': len(pairs),
            'similar_pairs': similar_count,
            'dissimilar_pairs': dissimilar_count,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'false_accept_rate': far,
            'false_reject_rate': frr,
            'avg_shouji_time_ms': avg_shouji_time,
            'avg_edlib_time_ms': avg_edlib_time,
            'speedup': avg_edlib_time / avg_shouji_time if avg_shouji_time > 0 else 0
        }
        
        print(f"\nResults:")
        print(f"  False Accept Rate: {far:.4f}%")
        print(f"  False Reject Rate: {frr:.4f}%")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/real_data_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("REAL DATA EXPERIMENTS COMPLETE")
    print("="*70)
    print("Results saved to results/real_data_results.json")
    
    return results


if __name__ == '__main__':
    run_real_data_experiments()
