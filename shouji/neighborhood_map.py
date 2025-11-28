"""
Neighborhood map utilities
"""

import numpy as np


class NeighborhoodMap:
    """
    Helper class for building and analyzing neighborhood maps
    """
    
    @staticmethod
    def build(text: str, pattern: str, edit_threshold: int) -> np.ndarray:
        """
        Build complete neighborhood map (all diagonals).
        
        Args:
            text: Reference sequence
            pattern: Query sequence  
            edit_threshold: Maximum edit distance
            
        Returns:
            2D numpy array of matches (0) and mismatches (1)
        """
        m = len(text)
        num_diagonals = 2 * edit_threshold + 1
        N = np.ones((num_diagonals, m), dtype=np.int8)
        
        for diag_offset in range(-edit_threshold, edit_threshold + 1):
            diag_idx = diag_offset + edit_threshold
            
            for i in range(m):
                j = i + diag_offset
                if 0 <= j < m:
                    N[diag_idx, i] = 0 if pattern[i] == text[j] else 1
        
        return N
    
    @staticmethod
    def visualize(N: np.ndarray) -> str:
        """
        Create string representation of neighborhood map.
        
        Args:
            N: Neighborhood map array
            
        Returns:
            String visualization
        """
        lines = []
        for i, row in enumerate(N):
            line = f"Diagonal {i}: " + "".join(str(x) for x in row)
            lines.append(line)
        return "\n".join(lines)
    
    @staticmethod
    def count_consecutive_zeros(vector: np.ndarray) -> int:
        """
        Count longest streak of consecutive zeros.
        
        Args:
            vector: 1D bit vector
            
        Returns:
            Length of longest zero streak
        """
        max_streak = 0
        current_streak = 0
        
        for bit in vector:
            if bit == 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak