import numpy as np
from numba import njit

@njit
def commonline_R(Ri, Rj, L):
    """
    Compute the common line induced by rotation matrices Ri and Rj.

    Returns the indices of the common lines between images (rotations) i and
    j in image i (l_ij) and in image j (l_ji), respectively.
    
    Parameters:
    Ri : np.ndarray
        3x3 rotation matrix
    Rj : np.ndarray
        3x3 rotation matrix
    L : int
        Number of discretized lines

    Returns:
    l_ij : int
        Common line index in image i
    l_ji : int
        Common line index in image j
    """
    Ut = np.dot(Rj, Ri.T)
    
    alphaij = np.arctan2(Ut[2, 0], -Ut[2, 1])
    alphaji = np.arctan2(-Ut[0, 2], Ut[1, 2])
    
    PI = 4 * np.arctan(1.0)
    alphaij += PI  # Shift from [-pi,pi] to [0,2*pi].
    alphaji += PI
    
    l_ij = int(np.round(alphaij / (2 * PI) * L) % L)
    l_ji = int(np.round(alphaji / (2 * PI) * L) % L)
    
    return l_ij, l_ji
