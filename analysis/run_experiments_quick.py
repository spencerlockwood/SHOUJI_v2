"""
Run experiments with smaller datasets for faster testing
"""

import os
import json
import time
import numpy as np
from typing import Dict, List
from tqdm import tqdm

from shouji import ShoujiFilter
from baseline import EdlibBaseline
from shouji.utils import generate_test_dataset


class QuickExperimentRunner:
    """Run experiments with smaller datasets"""
    
    def __init__(self, output_dir: str = 'results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def run_accuracy_experiment(self, seq_lengths: List[int], 
                               edit_thresholds: List[int],
                               num_pairs: int = 5000) -> Dict:  # Reduced from 30000
        """Run accuracy experiments"""
        results = {}
        
        for seq_length in seq_lengths:
            for edit_threshold in edit_thresholds:
                print(f"\n{'='*70}")
                print(f"Running: seq_length={seq_length}, edit_threshold={edit_threshold}")
                print(f"{'='*70}")
                
                # Generate dataset
                print(f"Generating {num_pairs} test pairs...")
                dataset = generate_test_dataset(num_pairs, seq_length, edit_threshold)
                
                # Initialize filters
                shouji = ShoujiFilter(edit_threshold, seq_length)
                edlib = EdlibBaseline(edit_threshold)
                
                # Collect results
                true_pos = 0
                true_neg = 0
                false_pos = 0
                false_neg = 0
                
                shouji_times = []
                edlib_times = []
                
                print("Processing sequence pairs...")
                for text, pattern, true_dist in tqdm(dataset):
                    # Edlib (ground truth)
                    start = time.time()
                    edlib_similar, edlib_dist = edlib.align(text, pattern)
                    edlib_times.append(time.time() - start)
                    
                    # Shouji
                    start = time.time()
                    shouji_similar, shouji_est = shouji.filter(text, pattern)
                    shouji_times.append(time.time() - start)
                    
                    # Compare
                    if edlib_similar:
                        if shouji_similar:
                            true_pos += 1
                        else:
                            false_neg += 1
                    else:
                        if shouji_similar:
                            false_pos += 1
                        else:
                            true_neg += 1
                
                # Calculate metrics
                similar_count = true_pos + false_neg
                dissimilar_count = false_pos + true_neg
                
                false_accept_rate = (false_pos / dissimilar_count * 100) if dissimilar_count > 0 else 0
                false_reject_rate = (false_neg / similar_count * 100) if similar_count > 0 else 0
                
                avg_shouji_time = np.mean(shouji_times) * 1000
                avg_edlib_time = np.mean(edlib_times) * 1000
                
                config_key = f"L{seq_length}_E{edit_threshold}"
                results[config_key] = {
                    'seq_length': seq_length,
                    'edit_threshold': edit_threshold,
                    'num_pairs': num_pairs,
                    'similar_pairs': similar_count,
                    'dissimilar_pairs': dissimilar_count,
                    'true_positives': true_pos,
                    'true_negatives': true_neg,
                    'false_positives': false_pos,
                    'false_negatives': false_neg,
                    'false_accept_rate': false_accept_rate,
                    'false_reject_rate': false_reject_rate,
                    'avg_shouji_time_ms': avg_shouji_time,
                    'avg_edlib_time_ms': avg_edlib_time,
                    'speedup': avg_edlib_time / avg_shouji_time if avg_shouji_time > 0 else 0
                }
                
                print(f"\nResults for {config_key}:")
                print(f"  Similar pairs: {similar_count}")
                print(f"  Dissimilar pairs: {dissimilar_count}")
                print(f"  False Accept Rate: {false_accept_rate:.4f}%")
                print(f"  False Reject Rate: {false_reject_rate:.4f}%")
                print(f"  Avg Shouji Time: {avg_shouji_time:.4f} ms")
                print(f"  Avg Edlib Time: {avg_edlib_time:.4f} ms")
        
        # Save results
        output_file = os.path.join(self.output_dir, 'accuracy_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
        return results
    
    def run_scalability_experiment(self, base_seq_length: int = 100,
                                  edit_threshold: int = 2,
                                  num_pairs: int = 2000) -> Dict:  # Reduced
        """Test scalability"""
        results = {}
        
        # Test E from 0 to 10
        edit_thresholds = list(range(0, 11))
        
        for E in edit_thresholds:
            print(f"\nTesting E={E}")
            
            dataset = generate_test_dataset(num_pairs, base_seq_length, E)
            shouji = ShoujiFilter(E, base_seq_length)
            edlib = EdlibBaseline(E)
            
            false_pos = 0
            false_neg = 0
            similar_count = 0
            dissimilar_count = 0
            
            for text, pattern, true_dist in tqdm(dataset):
                edlib_similar, _ = edlib.align(text, pattern)
                shouji_similar, _ = shouji.filter(text, pattern)
                
                if edlib_similar:
                    similar_count += 1
                    if not shouji_similar:
                        false_neg += 1
                else:
                    dissimilar_count += 1
                    if shouji_similar:
                        false_pos += 1
            
            false_accept_rate = (false_pos / dissimilar_count * 100) if dissimilar_count > 0 else 0
            false_reject_rate = (false_neg / similar_count * 100) if similar_count > 0 else 0
            
            results[f"E{E}"] = {
                'edit_threshold': E,
                'false_accept_rate': false_accept_rate,
                'false_reject_rate': false_reject_rate,
                'similar_pairs': similar_count,
                'dissimilar_pairs': dissimilar_count
            }
        
        # Save results
        output_file = os.path.join(self.output_dir, 'scalability_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def main():
    """Run all experiments"""
    
    runner = QuickExperimentRunner(output_dir='results')
    
    print("="*70)
    print("SHOUJI REIMPLEMENTATION - QUICK EXPERIMENTS")
    print("Using smaller datasets for faster execution")
    print("="*70)
    
    # Experiment 1: Accuracy across different configurations
    print("\n\n### EXPERIMENT 1: Accuracy Evaluation ###\n")
    seq_lengths = [100, 150, 250]
    edit_thresholds = [2, 5]
    accuracy_results = runner.run_accuracy_experiment(
        seq_lengths=seq_lengths,
        edit_thresholds=edit_thresholds,
        num_pairs=5000  # Reduced for speed
    )
    
    # Experiment 2: Scalability
    print("\n\n### EXPERIMENT 2: Scalability Analysis ###\n")
    scalability_results = runner.run_scalability_experiment(
        base_seq_length=100,
        edit_threshold=2,
        num_pairs=2000  # Reduced for speed
    )
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print("\nResults saved in 'results/' directory")
    print("Run 'python analysis/generate_plots.py' to create visualizations")


if __name__ == '__main__':
    main()
