"""
Edlib baseline for ground truth alignment
"""

import edlib
from typing import Tuple, List


class EdlibBaseline:
    """
    Wrapper for Edlib aligner to generate ground truth results
    """
    
    def __init__(self, edit_threshold: int):
        """
        Initialize Edlib baseline.
        
        Args:
            edit_threshold: Maximum edit distance
        """
        self.edit_threshold = edit_threshold
    
    def align(self, text: str, pattern: str) -> Tuple[bool, int]:
        """
        Align two sequences using Edlib.
        
        Args:
            text: Reference sequence
            pattern: Query sequence
            
        Returns:
            (is_similar, edit_distance): Whether within threshold and actual distance
        """
        result = edlib.align(pattern, text, task='distance', k=self.edit_threshold)
        
        edit_distance = result['editDistance']
        
        # editDistance is -1 if distance > k
        if edit_distance == -1:
            is_similar = False
            # Actual distance unknown, just set to threshold + 1
            edit_distance = self.edit_threshold + 1
        else:
            is_similar = edit_distance <= self.edit_threshold
        
        return is_similar, edit_distance
    
    def align_batch(self, sequence_pairs: List[Tuple[str, str]]) -> List[Tuple[bool, int]]:
        """
        Align multiple sequence pairs.
        
        Args:
            sequence_pairs: List of (text, pattern) tuples
            
        Returns:
            List of (is_similar, edit_distance) tuples
        """
        results = []
        for text, pattern in sequence_pairs:
            result = self.align(text, pattern)
            results.append(result)
        return results