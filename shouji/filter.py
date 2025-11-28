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
        
        return is_similar, num_edits
    
    def _build_neighborhood_map(self, text: str, pattern: str) -> np.ndarray:
        """
        Build neighborhood map showing matches/mismatches within edit threshold.
        
        Returns:
            2D numpy array where N[i,j] = 0 if match, 1 if mismatch
            Only diagonals within distance E from main diagonal are computed
        """
        # Create map for 2E+1 diagonals
        num_diagonals = 2 * self.E + 1
        neighborhood_map = np.ones((num_diagonals, self.m), dtype=np.int8)
        
        # Index mapping: diagonal k corresponds to map row k
        # k=0 is lower diagonal -E, k=E is main diagonal, k=2E is upper diagonal +E
        
        for diag_offset in range(-self.E, self.E + 1):
            diag_idx = diag_offset + self.E  # Map to array index
            
            for i in range(self.m):
                j = i + diag_offset
                if 0 <= j < self.m:
                    # Compare pattern[i] with text[j]
                    if pattern[i] == text[j]:
                        neighborhood_map[diag_idx, i] = 0
                    else:
                        neighborhood_map[diag_idx, i] = 1
        
        return neighborhood_map
    
    def _find_common_subsequences(self, neighborhood_map: np.ndarray) -> np.ndarray:
        """
        Use sliding window approach to find common subsequences.
        
        Args:
            neighborhood_map: 2D array of matches/mismatches
            
        Returns:
            Shouji bit-vector indicating edits along sequence
        """
        shouji_bitvector = np.ones(self.m, dtype=np.int8)
        num_diagonals = 2 * self.E + 1
        
        # Slide window across sequence
        for window_start in range(self.m):
            window_end = min(window_start + self.window_size, self.m)
            actual_window_size = window_end - window_start
            
            if actual_window_size < self.window_size:
                # Handle edge case at end of sequence
                continue
            
            # Find best 4-bit vector in this window
            best_vector = None
            best_zero_count = -1
            best_diag = -1
            
            # Check each diagonal
            for diag_idx in range(num_diagonals):
                # Extract 4-bit vector from this diagonal
                vector = neighborhood_map[diag_idx, window_start:window_end]
                zero_count = np.sum(vector == 0)
                
                # Tie-breaking: prefer vectors with leading zero
                if zero_count > best_zero_count or \
                   (zero_count == best_zero_count and vector[0] == 0 and 
                    (best_vector is None or best_vector[0] != 0)):
                    best_zero_count = zero_count
                    best_vector = vector
                    best_diag = diag_idx
            
            # Update Shouji bit-vector if this improves it
            if best_vector is not None:
                current_zero_count = np.sum(
                    shouji_bitvector[window_start:window_end] == 0
                )
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