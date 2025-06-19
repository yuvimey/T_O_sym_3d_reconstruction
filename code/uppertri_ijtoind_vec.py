import numpy as np

def uppertri_ijtoind_vec(i, j, n):
    """
    Convert (i, j) subscripts (i < j) into linear indices for the upper 
    triangular part of an n x n matrix, excluding the diagonal.
    
    Example:
        (0,1) -> 0
        (n-1, n) -> n*(n-1)//2 - 1
    """
    ind = None
    if type(i) == int:
        i = i+1
        j = j+1
        
        ind = ((2 * n - i) * (i - 1)) // 2 + (j - i) - 1
    else:
        i = [ii+1 for ii in i]
        j = [jj+1 for jj in j]
        ind = [((2 * n - i[k]) * (i[k] - 1)) // 2 + (j[k] - i[k]) - 1 for k in range(len(i))]
    return ind



