import numpy as np
from scipy.spatial.transform import Rotation as R

def rand_rots(n):
    """
    Generate `n` random rotation matrices uniformly distributed in SO(3).
    
    Parameters:
        n (int): Number of random rotation matrices to generate.

    Returns:
        np.ndarray: Array of shape (3, 3, n) containing n random rotation matrices.
    
    Note:
        This function depends on the random state of the NumPy random generator, so to
        obtain consistent outputs, its state must be controlled using `np.random.seed()`.
    """
    # Generate `n` random quaternions
    qs = R.random(n).as_quat()  # Generates random quaternions
    '''for i in range(n):
        qs[i] = [0.0, 0.0, 0.0, 0.0]
        qs[i][i % 4] = (-1.0) ** (i//4)'''
    
    # Convert quaternions to rotation matrices
    qs[0]= [1.0, 0.0, 0.0, 0.0]
    rot_matrices = q_to_rot(qs)
    
    return rot_matrices, np.array(qs)

def q_to_rot(q):
    """
    Convert a quaternion into a rotation matrix.
    
    Parameters:
        q (np.ndarray): Quaternions of shape (n, 4).

    Returns:
        np.ndarray: Rotation matrices of shape (3, 3, n).
    """
    q = np.array(q)
    n = q.shape[0]
    rot_matrix = np.zeros((3, 3, n))
    
    rot_matrix[0, 0, :] = q[:, 0]**2 + q[:, 1]**2 - q[:, 2]**2 - q[:, 3]**2
    rot_matrix[0, 1, :] = 2 * q[:, 1] * q[:, 2] - 2 * q[:, 0] * q[:, 3]
    rot_matrix[0, 2, :] = 2 * q[:, 0] * q[:, 2] + 2 * q[:, 1] * q[:, 3]
    
    rot_matrix[1, 0, :] = 2 * q[:, 1] * q[:, 2] + 2 * q[:, 0] * q[:, 3]
    rot_matrix[1, 1, :] = q[:, 0]**2 - q[:, 1]**2 + q[:, 2]**2 - q[:, 3]**2
    rot_matrix[1, 2, :] = -2 * q[:, 0] * q[:, 1] + 2 * q[:, 2] * q[:, 3]
    
    rot_matrix[2, 0, :] = -2 * q[:, 0] * q[:, 2] + 2 * q[:, 1] * q[:, 3]
    rot_matrix[2, 1, :] = 2 * q[:, 0] * q[:, 1] + 2 * q[:, 2] * q[:, 3]
    rot_matrix[2, 2, :] = q[:, 0]**2 - q[:, 1]**2 - q[:, 2]**2 + q[:, 3]**2
    
    return rot_matrix