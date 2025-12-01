"""
Core Shouji filter implementation
"""

import numpy as np
from typing import Tuple


class ShoujiFilter:
    """
    Shouji pre-alignment filter based on pigeonhole principle.
    
    If two sequences differ by E edits, they must share at least one
    common subsequence with total length >= m - E.
    """
    
    def __init__(self, edit_threshold: int, sequence_length: int):
        """
        Initialize Shouji filter.
        
        Args:
            edit_threshold: Maximum allowed edit distance (E)
            sequence_length: Length of sequences to compare (m)
        """
        self.E = edit_threshold
        self.m = sequence_length
        self.window_size = 4  # As specified in paper
        self.step_size = 1
        
    def filter(self, text: str, pattern: str) -> Tuple[bool, int]:
        """
        Apply Shouji filter to determine if sequences are similar.
        
        Args:
            text: Reference sequence
            pattern: Query sequence
            
        Returns:
            (is_similar, estimated_edits): Whether sequences pass filter and estimated edit count
        """
        if len(text) != len(pattern) or len(text) != self.m:
            raise ValueError(f"Sequences must be length {self.m}")
        
        # Step 1: Build neighborhood map
        neighborhood_map = self._build_neighborhood_map(text, pattern)
        
        # Step 2: Identify diagonally consecutive matches
        shouji_bitvector = self._find_common_subsequences(neighborhood_map)
        
        # Step 3: Filter out dissimilar sequences
        num_edits = np.sum(shouji_bitvector)
        is_similar = num_edits <= self.E
        
        return is_similar, int(num_edits)
    
    def _build_neighborhood_map(self, text: str, pattern: str) -> np.ndarray:
        """
        Build neighborhood map showing matches/mismatches within edit threshold.
        
        The neighborhood map compares pattern[i] with text[j] where j is within
        E positions of i (representing possible insertions/deletions).
        
        Returns:
            2D numpy array where N[i,j] = 0 if match, 1 if mismatch
            Shape: (2E+1, m) representing diagonals
        """
        num_diagonals = 2 * self.E + 1
        neighborhood_map = np.ones((num_diagonals, self.m), dtype=np.int8)
        
        # For each diagonal offset from -E to +E
        for diag_offset in range(-self.E, self.E + 1):
            diag_idx = diag_offset + self.E  # Map to array index [0, 2E]
            
            # For each position along the pattern
            for i in range(self.m):
                j = i + diag_offset
                # Check if j is within bounds of text
                if 0 <= j < self.m:
                    # Compare pattern[i] with text[j]
                    if pattern[i] == text[j]:
                        neighborhood_map[diag_idx, i] = 0
        
        return neighborhood_map
    
    def _count_zeros(self, vector: np.ndarray) -> int:
        """Count number of zeros in a vector"""
        return int(np.sum(vector == 0))
    
    def _find_common_subsequences(self, neighborhood_map: np.ndarray) -> np.ndarray:
        """
        Use sliding window approach to find common subsequences.
        
        For each window position, we select the 4-bit diagonal vector with the
        maximum number of zeros (matches). We then update the Shouji bit-vector
        only if this improves the number of matches at that position.
        
        Args:
            neighborhood_map: 2D array of matches/mismatches [diagonals x positions]
            
        Returns:
            Shouji bit-vector indicating estimated edits along sequence
        """
        shouji_bitvector = np.ones(self.m, dtype=np.int8)
        num_diagonals = 2 * self.E + 1
        
        # Process each window position
        num_windows = self.m - self.window_size + 1
        
        for window_start in range(num_windows):
            window_end = window_start + self.window_size
            
            best_vector = None
            best_zero_count = -1
            best_has_leading_zero = False
            
            # Check all diagonals for this window
            for diag_idx in range(num_diagonals):
                vector = neighborhood_map[diag_idx, window_start:window_end]
                zero_count = self._count_zeros(vector)
                has_leading_zero = (vector[0] == 0)
                
                # Decide if this is the best vector so far
                is_better = False
                
                if zero_count > best_zero_count:
                    # More zeros is always better
                    is_better = True
                elif zero_count == best_zero_count and zero_count > 0:
                    # Tie-breaking: prefer vector with leading zero
                    if has_leading_zero and not best_has_leading_zero:
                        is_better = True
                
                if is_better:
                    best_vector = vector.copy()
                    best_zero_count = zero_count
                    best_has_leading_zero = has_leading_zero
            
            # Update Shouji bit-vector if this improves it
            if best_vector is not None:
                current_zero_count = self._count_zeros(
                    shouji_bitvector[window_start:window_end]
                )
                
                # Update if we found more matches (zeros) in this window
                if best_zero_count > current_zero_count:
                    shouji_bitvector[window_start:window_end] = best_vector
        
        return shouji_bitvector
    
    def filter_batch(self, sequence_pairs: list) -> list:
        """
        Filter multiple sequence pairs.
        
        Args:
            sequence_pairs: List of (text, pattern) tuples
            
        Returns:
            List of (is_similar, estimated_edits) tuples
        """
        results = []
        for text, pattern in sequence_pairs:
            result = self.filter(text, pattern)
            results.append(result)
        return results