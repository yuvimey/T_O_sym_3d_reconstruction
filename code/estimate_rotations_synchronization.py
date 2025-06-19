import numpy as np
from uppertri_ijtoind_vec import uppertri_ijtoind_vec
from multi_Jify import multi_Jify
from scipy.linalg import svd
from scipy.sparse.linalg import eigs
from numpy.linalg import eig, eigh

def estimate_rotations_synchronization(est_rel_rots, u_G, cache_file_name):
    # Estimates the rotations using synchronization method.

    # Load the candidates set 'R'
    data = np.load(cache_file_name, allow_pickle=True)
    R = data['R']

    n_images = est_rel_rots.shape[0]
    J = np.diag([1, 1, -1])

    H = np.zeros((3 * n_images, 3 * n_images, 3))

    # Create single-entry matrices
    e_kl = np.zeros((3, 3, 9))
    for idx in range(9):
        e_kl[:, :, idx][idx // 3, idx % 3] = 1

    # Construct block (1,2)
    pair_ind = uppertri_ijtoind_vec(0, 1, n_images)
    H[0:3, 3:6, 0] = R[:, :, est_rel_rots[0, 1]] @ e_kl[:, :, 0] @ R[:, :, est_rel_rots[1, 0]].T
    H[0:3, 3:6, 1] = R[:, :, est_rel_rots[0, 1]] @ e_kl[:, :, 4] @ R[:, :, est_rel_rots[1, 0]].T
    H[0:3, 3:6, 2] = R[:, :, est_rel_rots[0, 1]] @ e_kl[:, :, 8] @ R[:, :, est_rel_rots[1, 0]].T
    if u_G[pair_ind] < 0:
        H[0:3, 3:6, :] = multi_Jify(H[0:3, 3:6, :])

    # Construct blocks (1,j) for j=3..n_images
    for p_j in range(2, n_images):
        max_norm = [0, 0, 0]
        pair_ind = uppertri_ijtoind_vec(0, p_j, n_images)
        for kl in [0, 4, 8]:
            H1j = R[:, :, est_rel_rots[0, p_j]] @ e_kl[:, :, kl] @ R[:, :, est_rel_rots[p_j, 0]].T
            if u_G[pair_ind] < 0:
                H1j = J @ H1j @ J
            for m in range(3):
                norm_val = np.linalg.norm(H[0:3, 3:6, m].T @ H1j)
                if norm_val > max_norm[m]:
                    H[0:3, 3 * p_j:3 * (p_j + 1), m] = H1j
                    max_norm[m] = norm_val

    # Construct blocks (i,j) for i=2..n_images-1, j=i+1..n_images
    for p_i in range(1, n_images - 1):
        for p_j in range(p_i + 1, n_images):
            min_norm = [10, 10, 10]
            Hij_base = [
                np.copy(H[0:3, 3 * p_i:3 * (p_i + 1), m].T @ H[0:3, 3 * p_j:3 * (p_j + 1), m])
                for m in range(3)
            ]
            pair_ind = uppertri_ijtoind_vec(p_i, p_j, n_images)
            for kl in range(9):
                for sign in [1, -1]:
                    Hij = (
                        R[:, :, est_rel_rots[p_i, p_j]] @ (sign * e_kl[:, :, kl]) @ R[:, :, est_rel_rots[p_j, p_i]].T
                    )
                    if u_G[pair_ind] < 0:
                        Hij = np.copy(J @ Hij @ J)
                    for m in range(3):
                        norm_diff = np.linalg.norm(Hij_base[m] - Hij)
                        if norm_diff < min_norm[m]:
                            H[3 * p_i:3 * (p_i + 1), 3 * p_j:3 * (p_j + 1), m] = np.copy(Hij)
                            min_norm[m] = norm_diff

    # Diagonal blocks (i, i)
    for m in range(3):
        for p_i in range(n_images):
            for p_j in range(n_images):
                if p_i < p_j:
                    H[3 * p_i:3 * (p_i + 1), 3 * p_i:3 * (p_i + 1), m] += (
                        H[3 * p_i:3 * (p_i + 1), 3 * p_j:3 * (p_j + 1), m] @
                        H[3 * p_i:3 * (p_i + 1), 3 * p_j:3 * (p_j + 1), m].T
                    )
                elif p_i > p_j:
                    H[3 * p_i:3 * (p_i + 1), 3 * p_i:3 * (p_i + 1), m] += (
                        H[3 * p_j:3 * (p_j + 1), 3 * p_i:3 * (p_i + 1), m].T @
                        H[3 * p_j:3 * (p_j + 1), 3 * p_i:3 * (p_i + 1), m]
                    )
            Hii = np.copy(H[3 * p_i:3 * (p_i + 1), 3 * p_i:3 * (p_i + 1), m] / (2 * (n_images - 1)))
            u, s, v = svd(Hii, lapack_driver='gesvd')
            v = v.T
            idx = np.argmax(s)
            H[3 * p_i:3 * (p_i + 1), 3 * p_i:3 * (p_i + 1), m] = s[idx] * np.outer(u[:, idx], v[:, idx])

    for m in range(3):
        H[:, :, m] = H[:, :, m] + H[:, :, m].T

    # Synchronize using SVD decomposition
    V = np.zeros((3 * n_images, 3))
    for m in range(3):
        k = min(20,H[:, :, m].shape[0] - 2)

        vals, vecs = eigh(np.copy(H[:, :, m]))
        idx = np.argsort(vals)[::-1]
        evect1 = vecs[:, idx[0]].real

        for i in range(n_images):
            vi = np.copy(evect1[3 * i:3 * (i + 1)])
            vi = vi / np.linalg.norm(vi)
            V[3 * i:3 * (i + 1), m] = vi

    rots = np.stack([
        V[:, 0].reshape(3, n_images, order='F'),
        V[:, 1].reshape(3, n_images, order='F'),
        V[:, 2].reshape(3, n_images, order='F')
    ])

    if np.linalg.det(rots[:, :, 0]) < 0:
        rots = -rots

    return rots, H