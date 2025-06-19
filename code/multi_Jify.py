import numpy as np

def multi_Jify(rotations):
    """
    Applies J * R * J where J = diag(1, 1, -1) to a batch of 3x3 rotation matrices.

    Parameters:
        rotations: np.ndarray of shape (3, 3, ...) – can be (3, 3, N) or higher dimensional

    Returns:
        Jified_rot: np.ndarray of the same shape as input, transformed
    """
    dims = rotations.shape
    Jified_rot = rotations.reshape(9, *dims[2:], order='F')  # flatten 3x3 to vector

    # Indices corresponding to elements affected by J * R * J
    # These correspond to matrix indices (3,1), (2,3), (1,3), (1,2) — 0-based indexing:
    affected_indices = [2, 5, 6, 7]
    Jified_rot[affected_indices, ...] *= -1

    return Jified_rot.reshape(3, 3, *dims[2:], order='F')