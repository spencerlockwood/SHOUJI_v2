from shouji import ShoujiFilter
import numpy as np

# Paper example from Figure 1
text = "GGTGCAGAGCTC"
pattern = "GGTGAGAGTTGT"
seq_length = len(text)
edit_threshold = 3

print(f"Text:    {text}")
print(f"Pattern: {pattern}")
print(f"Length: {seq_length}, E: {edit_threshold}")
print()

shouji = ShoujiFilter(edit_threshold, seq_length)

# Build neighborhood map
neighborhood_map = shouji._build_neighborhood_map(text, pattern)

print("Neighborhood Map:")
print("Diagonal indices (0=lower E, E=main, 2E=upper E)")
for i, row in enumerate(neighborhood_map):
    diag_offset = i - edit_threshold
    diag_label = f"Diagonal {diag_offset:+2d}"
    print(f"{diag_label}: {''.join(str(x) for x in row)}")

print()

# Get Shouji bit-vector
shouji_bitvector = shouji._find_common_subsequences(neighborhood_map)
print(f"Shouji bit-vector: {''.join(str(x) for x in shouji_bitvector)}")
print(f"Number of edits (1s): {np.sum(shouji_bitvector)}")
print(f"Number of matches (0s): {np.sum(shouji_bitvector == 0)}")

# Filter result
is_similar, num_edits = shouji.filter(text, pattern)
print(f"\nResult: is_similar={is_similar}, num_edits={num_edits}")
print(f"Expected: is_similar=True (3 edits <= threshold 3)")

# Compare character by character for main diagonal
print("\nCharacter-by-character comparison:")
for i in range(seq_length):
    match = "✓" if text[i] == pattern[i] else "✗"
    print(f"Position {i:2d}: text[{text[i]}] vs pattern[{pattern[i]}] {match}")
