import numpy as np
from scipy.special import comb

def lin2sub3_map(N):
    """
    Generate a mapping of linear indices to 3D subscript indices for an N-choose-3 combination.
    
    Parameters:
    - N: int
        The number of elements to choose from.
    
    Returns:
    - idx_map: numpy.ndarray
        An array of shape (N-choose-3, 3) containing the index mappings.
    """
    idx_map = np.zeros((int(comb(N, 3)), 3), dtype=int)
    idx = 0
    
    for i in range(N-2):
        for j in range(i + 1, N-1):
            for k in range(j + 1, N):
                idx_map[idx, :] = [i, j, k]
                idx += 1
    
    return idx_map