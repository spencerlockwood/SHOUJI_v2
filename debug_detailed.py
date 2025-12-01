from shouji import ShoujiFilter
import numpy as np

text = "GGTGCAGAGCTC"
pattern = "GGTGAGAGTTGT"
seq_length = len(text)
edit_threshold = 3

print(f"Text:    {text}")
print(f"Pattern: {pattern}")
print(f"E = {edit_threshold}, m = {seq_length}")
print()

# Manual alignment to understand the true edit distance
print("Expected alignment (from paper Figure 1):")
print("The paper shows 3 common subsequences:")
print("  1. GGTG (positions 0-3)")
print("  2. AGAG (around positions 4-7, shifted)")
print("  3. T (around position 10, shifted)")
print("Total matches should be >= m - E = 12 - 3 = 9")
print()

shouji = ShoujiFilter(edit_threshold, seq_length)
neighborhood_map = shouji._build_neighborhood_map(text, pattern)

print("="*60)
print("NEIGHBORHOOD MAP ANALYSIS")
print("="*60)

# Show each diagonal
for diag_idx in range(2 * edit_threshold + 1):
    offset = diag_idx - edit_threshold
    row = neighborhood_map[diag_idx]
    zeros = np.sum(row == 0)
    print(f"Diagonal {offset:+2d}: {''.join(str(x) for x in row)} ({zeros} matches)")

print()
print("="*60)
print("SLIDING WINDOW ANALYSIS")
print("="*60)

# Manually process windows to see what's selected
for window_start in range(seq_length - 4 + 1):
    window_end = window_start + 4
    print(f"\nWindow [{window_start}:{window_end}]:")
    
    best_diag = None
    best_zeros = -1
    
    for diag_idx in range(2 * edit_threshold + 1):
        offset = diag_idx - edit_threshold
        vector = neighborhood_map[diag_idx, window_start:window_end]
        zeros = np.sum(vector == 0)
        
        if zeros > best_zeros or (zeros == best_zeros and vector[0] == 0):
            if zeros > best_zeros:
                best_diag = offset
                best_zeros = zeros
            elif vector[0] == 0:
                best_diag = offset
        
        if zeros > 0:
            print(f"  Diag {offset:+2d}: {''.join(str(x) for x in vector)} ({zeros} zeros)")
    
    print(f"  → Best: Diagonal {best_diag:+2d} with {best_zeros} zeros")

print()
print("="*60)
print("FINAL RESULT")
print("="*60)

shouji_bitvector = shouji._find_common_subsequences(neighborhood_map)
print(f"Shouji bit-vector: {''.join(str(x) for x in shouji_bitvector)}")
print(f"Edits found: {np.sum(shouji_bitvector)}")
print(f"Matches found: {np.sum(shouji_bitvector == 0)}")
print(f"Required matches: >= {seq_length - edit_threshold}")

is_similar, num_edits = shouji.filter(text, pattern)
print(f"\nFilter result: {is_similar} (edits={num_edits}, threshold={edit_threshold})")
print(f"Expected: True (should accept)")

# Calculate actual edit distance using simple algorithm
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
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

true_ed = edit_distance(text, pattern)
print(f"\nTrue edit distance (Levenshtein): {true_ed}")
