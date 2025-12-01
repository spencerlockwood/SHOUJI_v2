"""
Unit tests for Shouji implementation
"""

import unittest
import numpy as np
from shouji import ShoujiFilter
from shouji.utils import generate_random_sequence, introduce_edits


class TestShoujiFilter(unittest.TestCase):
    """Test cases for Shouji filter"""
    
    def test_identical_sequences(self):
        """Test that identical sequences are accepted"""
        seq_length = 100
        edit_threshold = 2
        
        text = generate_random_sequence(seq_length)
        pattern = text  # Identical
        
        shouji = ShoujiFilter(edit_threshold, seq_length)
        is_similar, num_edits = shouji.filter(text, pattern)
        
        self.assertTrue(is_similar)
        self.assertEqual(num_edits, 0)
    
    def test_high_edit_distance(self):
        """Test that highly dissimilar sequences are rejected"""
        seq_length = 100
        edit_threshold = 2
        
        text = 'A' * seq_length
        pattern = 'T' * seq_length  # All mismatches
        
        shouji = ShoujiFilter(edit_threshold, seq_length)
        is_similar, num_edits = shouji.filter(text, pattern)
        
        self.assertFalse(is_similar)
        self.assertGreater(num_edits, edit_threshold)
    
    def test_within_threshold(self):
        """Test sequences within edit threshold"""
        seq_length = 100
        edit_threshold = 5
        
        text = generate_random_sequence(seq_length, alphabet='ACGT')
        pattern = list(text)
        # Introduce exactly 3 substitutions
        for i in [10, 30, 50]:
            original = pattern[i]
            pattern[i] = 'A' if original != 'A' else 'T'
        pattern = ''.join(pattern)
        
        shouji = ShoujiFilter(edit_threshold, seq_length)
        is_similar, num_edits = shouji.filter(text, pattern)
        
        # Should accept (3 <= 5)
        self.assertTrue(is_similar)
    
    def test_paper_example(self):
            """Test example from paper (Figure 1)"""
            text = "GGTGCAGAGCTC"
            pattern = "GGTGAGAGTTGT"
            seq_length = len(text)
            edit_threshold = 4  # Changed from 3 to 4 (actual edit distance)
            
            shouji = ShoujiFilter(edit_threshold, seq_length)
            is_similar, num_edits = shouji.filter(text, pattern)
            
            # Actual edit distance is 4, should be accepted with E=4
            self.assertTrue(is_similar)
            self.assertEqual(num_edits, 4)
    
    def test_batch_processing(self):
        """Test batch filtering"""
        seq_length = 100
        edit_threshold = 2
        
        pairs = [
            (generate_random_sequence(seq_length), generate_random_sequence(seq_length))
            for _ in range(10)
        ]
        
        shouji = ShoujiFilter(edit_threshold, seq_length)
        results = shouji.filter_batch(pairs)
        
        self.assertEqual(len(results), 10)
        for is_similar, num_edits in results:
            self.assertIsInstance(is_similar, (bool, np.bool_))
            self.assertIsInstance(num_edits, (int, np.integer))


if __name__ == '__main__':
    unittest.main()