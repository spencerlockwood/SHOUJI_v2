"""
Validate Shouji accuracy against Edlib baseline
"""

import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

from shouji import ShoujiFilter
from baseline import EdlibBaseline
from shouji.utils import generate_test_dataset


class AccuracyValidator:
    """Validate filter accuracy"""
    
    def __init__(self, edit_threshold: int, seq_length: int):
        self.edit_threshold = edit_threshold
        self.seq_length = seq_length
        self.shouji = ShoujiFilter(edit_threshold, seq_length)
        self.edlib = EdlibBaseline(edit_threshold)
    
    def validate(self, dataset: List[Tuple[str, str, int]]) -> Dict:
        """
        Validate Shouji against ground truth.
        
        Args:
            dataset: List of (text, pattern, true_edit_distance) tuples
            
        Returns:
            Dictionary with accuracy metrics
        """
        true_positives = 0  # Correctly accepted similar pairs
        true_negatives = 0  # Correctly rejected dissimilar pairs
        false_positives = 0  # Incorrectly accepted (false accept)
        false_negatives = 0  # Incorrectly rejected (false reject)
        
        shouji_results = []
        edlib_results = []
        
        print(f"Validating {len(dataset)} sequence pairs...")
        
        for text, pattern, true_dist in tqdm(dataset):
            # Get ground truth from Edlib
            edlib_similar, edlib_dist = self.edlib.align(text, pattern)
            
            # Get Shouji prediction
            shouji_similar, shouji_est = self.shouji.filter(text, pattern)
            
            shouji_results.append((shouji_similar, shouji_est))
            edlib_results.append((edlib_similar, edlib_dist))
            
            # Compare against ground truth
            if edlib_similar:  # Ground truth: similar
                if shouji_similar:
                    true_positives += 1
                else:
                    false_negatives += 1  # FALSE REJECT (bad!)
            else:  # Ground truth: dissimilar
                if shouji_similar:
                    false_positives += 1  # FALSE ACCEPT
                else:
                    true_negatives += 1
        
        # Calculate metrics
        total = len(dataset)
        similar_count = sum(1 for edlib_sim, _ in edlib_results if edlib_sim)
        dissimilar_count = total - similar_count
        
        false_accept_rate = (false_positives / dissimilar_count * 100) if dissimilar_count > 0 else 0
        false_reject_rate = (false_negatives / similar_count * 100) if similar_count > 0 else 0
        
        accuracy = (true_positives + true_negatives) / total * 100
        
        results = {
            'total_pairs': total,
            'similar_pairs': similar_count,
            'dissimilar_pairs': dissimilar_count,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'false_accept_rate': false_accept_rate,
            'false_reject_rate': false_reject_rate,
            'accuracy': accuracy,
            'shouji_results': shouji_results,
            'edlib_results': edlib_results
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print validation results"""
        print("\n" + "="*60)
        print("SHOUJI VALIDATION RESULTS")
        print("="*60)
        print(f"Total sequence pairs:     {results['total_pairs']}")
        print(f"Similar pairs (≤E):       {results['similar_pairs']}")
        print(f"Dissimilar pairs (>E):    {results['dissimilar_pairs']}")
        print(f"\nTrue Positives:           {results['true_positives']}")
        print(f"True Negatives:           {results['true_negatives']}")
        print(f"False Positives:          {results['false_positives']}")
        print(f"False Negatives:          {results['false_negatives']}")
        print(f"\nFalse Accept Rate:        {results['false_accept_rate']:.4f}%")
        print(f"False Reject Rate:        {results['false_reject_rate']:.4f}%")
        print(f"Overall Accuracy:         {results['accuracy']:.2f}%")
        print("="*60)


def main():
    """Run validation tests"""
    
    # Test configurations matching paper
    configs = [
        {'seq_length': 100, 'edit_threshold': 2, 'num_pairs': 10000},
        {'seq_length': 100, 'edit_threshold': 5, 'num_pairs': 10000},
    ]
    
    all_results = {}
    
    for config in configs:
        seq_length = config['seq_length']
        edit_threshold = config['edit_threshold']
        num_pairs = config['num_pairs']
        
        print(f"\n{'='*60}")
        print(f"Testing: Length={seq_length}, E={edit_threshold}")
        print(f"{'='*60}")
        
        # Generate test dataset
        dataset = generate_test_dataset(num_pairs, seq_length, edit_threshold)
        
        # Validate
        validator = AccuracyValidator(edit_threshold, seq_length)
        results = validator.validate(dataset)
        validator.print_results(results)
        
        # Store results
        key = f"L{seq_length}_E{edit_threshold}"
        all_results[key] = results
    
    return all_results


if __name__ == '__main__':
    main()