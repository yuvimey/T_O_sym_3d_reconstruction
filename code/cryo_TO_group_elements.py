import numpy as np

def cryo_TO_group_elements(symmetry):
    """
    Defines the symmetry group elements for Tetrahedral ('T') or Octahedral ('O') symmetries.

    Parameters:
        symmetry (str): Either 'T' or 'O'.

    Returns:
        gR (np.ndarray): Array of 3x3 symmetry group rotation matrices.
        scl_inds (list): Indices of self common line pairs.
    """
    if symmetry == 'T':
        n_gR = 12
        scl_inds = [1, 3, 5, 7, 9, 10, 11]

        gR = np.zeros((3, 3, n_gR))
        gR[:, :, 0] = np.eye(3)
        gR[:, :, 1] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        gR[:, :, 2] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        gR[:, :, 3] = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
        gR[:, :, 4] = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
        gR[:, :, 5] = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        gR[:, :, 6] = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
        gR[:, :, 7] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        gR[:, :, 8] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        gR[:, :, 9] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        gR[:, :, 10] = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        gR[:, :, 11] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    elif symmetry == 'O':
        n_gR = 24
        scl_inds = [1, 3, 5, 7, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23]

        gR = np.zeros((3, 3, n_gR))
        gR[:, :, 0] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        gR[:, :, 1] = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        gR[:, :, 2] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        gR[:, :, 3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        gR[:, :, 4] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        gR[:, :, 5] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        gR[:, :, 6] = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
        gR[:, :, 7] = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        gR[:, :, 8] = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        gR[:, :, 9] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        gR[:, :, 10] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        gR[:, :, 11] = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        gR[:, :, 12] = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
        gR[:, :, 13] = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
        gR[:, :, 14] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        gR[:, :, 15] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        gR[:, :, 16] = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
        gR[:, :, 17] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        gR[:, :, 18] = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
        gR[:, :, 19] = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        gR[:, :, 20] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        gR[:, :, 21] = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
        gR[:, :, 22] = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
        gR[:, :, 23] = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

    else:
        raise ValueError("Wrong symmetry type: should be 'T' or 'O'.")

    return gR, scl_inds