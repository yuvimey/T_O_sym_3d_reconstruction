import numpy as np
from cryo_TO_group_elements import cryo_TO_group_elements
from cryo_TO_configuration import cryo_TO_configuration
from genRotationsGrid import genRotationsGrid
from commonline_R import commonline_R
from numba import njit
import math

def cryo_TO_create_cache(symmetry):
    """
    Creates a cache file containing:
        - Candidate rotation matrices.
        - Common lines indices.
        - Self common lines indices.
    
    Parameters:
        symmetry: str
            Either 'T' for tetrahedral symmetry or 'O' for octahedral symmetry.
    
    Returns:
        cache_filename: str
            The name of the cache file created.
    """
    n_theta, _, _, resolution, viewing_angle, inplane_rot_degree = cryo_TO_configuration()
    gR, scl_inds = cryo_TO_group_elements(symmetry)
    
    print('Generating candidate rotations set')
    R = TO_candidate_rotations_set(gR, resolution, viewing_angle, inplane_rot_degree)
    
    n_candidates = R.shape[2]
    print('Computing common lines and self common lines indices sets')
    l_self_ind, l_ij_ji_ind = TO_cl_scl_inds(gR, np.asarray(scl_inds), n_theta, R)
    
    cache_filename = f'{symmetry}_symmetry_{n_candidates}_candidates_cache.mat'
    print(f'Cache file name: {cache_filename}')
    
    np.savez(cache_filename, l_self_ind=l_self_ind, l_ij_ji_ind=l_ij_ji_ind, R=R, n_theta=n_theta)
    
    return cache_filename + str(".npz")

def TO_candidate_rotations_set(gR, resolution, viewing_angle, inplane_rot_degree):
    """
    Generates approximately equally spaced rotations and filters rotations
    with close viewing angle and close in-plane rotation based on symmetry.
    """
    n_gR = gR.shape[2]
    
    candidates_set = genRotationsGrid(resolution)[0]
    n_candidates = candidates_set.shape[2]

    candidates_set = np.ascontiguousarray(candidates_set)
    
    close_idx = np.zeros(n_candidates, dtype=bool)

    print("n_candidates: " + str(n_candidates) + " n_gR: " + str(n_gR))
    @njit 
    def __TO_candidate_rotations_set(gR, close_idx, candidates_set):
        for r_i in range(n_candidates - 1):
            print(f"     TO_candidate_rotations_set Iter ({r_i}/{n_candidates - 1})")
            if close_idx[r_i]:
                continue
            Ri = candidates_set[:, :, r_i]
            for r_j in range(r_i + 1, n_candidates):
                if close_idx[r_j]:
                    continue
                Rj = candidates_set[:, :, r_j]
                for k in range(n_gR):
                    gRj = gR[:, :, k] @ Rj
                    if np.sum(Ri[:, 2] * gRj[:, 2]) > viewing_angle:
                        R_inplane = Ri.T @ gRj
                        theta = abs(math.degrees(np.arctan2(R_inplane[1, 0], R_inplane[0, 0])))
                        if theta < inplane_rot_degree:
                            close_idx[r_j] = True
    
        return candidates_set[:, :, ~close_idx]
    return __TO_candidate_rotations_set(gR, close_idx, candidates_set)


def TO_cl_scl_inds(gR, scl_inds, n_theta, R):
    """
    Computes common line indices and self common line indices from rotation matrices and group elements.

    Parameters:
    - gR: ndarray of shape (3, 3, group_order), symmetry group elements
    - scl_inds: ndarray of shape (n_scl_pairs), indices for self common lines
    - n_theta: int, number of radial lines
    - R: ndarray of shape (3, 3, n_R), rotation matrices

    Returns:
    - l_self_ind: ndarray of shape (n_R, n_scl_pairs), self common line indices (linear)
    - l_ij_ji_ind: ndarray of shape (n_R * n_R, n_gR), pairwise common line indices (linear)
    """
    n_R = R.shape[2]
    n_gR = gR.shape[2]
    n_scl_pairs = scl_inds.shape[0]

    l_self = np.zeros((2, n_scl_pairs, n_R), dtype=int)
    l_ij_ji = np.zeros((2, n_gR, n_R * n_R), dtype=int)
    @njit
    def __TO_cl_scl_inds(l_self, l_ij_ji, gR, scl_inds, n_theta, R, n_R, n_gR, n_scl_pairs):
        for i in range(n_R):
            print(f"     TO_cl_scl_inds Iter ({i}/{n_R}) ")
            # Compute self common lines for R_i
            for k in range(n_scl_pairs):
                l1, l2 = commonline_R(R[:, :, i], R[:, :, i] @ gR[:, :, scl_inds[k]], n_theta)
                l_self[0, k, i] = l1
                l_self[1, k, i] = l2

            for j in range(n_R):
                if i == j:
                    continue

                ind = j * n_R + i
                for k in range(n_gR):
                    l1, l2 = commonline_R(R[:, :, i], R[:, :, j] @ gR[:, :, k], n_theta)
                    l_ij_ji[0, k, ind] = l1
                    l_ij_ji[1, k, ind] = l2
        return l_self, l_ij_ji
    
    l_self, l_ij_ji = __TO_cl_scl_inds(l_self, l_ij_ji, gR, scl_inds, n_theta, R, n_R, n_gR, n_scl_pairs)
    # Convert to linear indices
    l_self_ind = np.ravel_multi_index((l_self[0], l_self[1]), dims=(n_theta, n_theta),order='F')
    l_ij_ji_ind = np.ravel_multi_index((l_ij_ji[0], l_ij_ji[1]), dims=(n_theta, n_theta), order='F')

    # Reshape
    l_self_ind = l_self_ind.reshape(n_scl_pairs, n_R, order='F').T
    l_ij_ji_ind = l_ij_ji_ind.reshape(n_gR, n_R * n_R, order='F').T

    return l_self_ind, l_ij_ji_ind