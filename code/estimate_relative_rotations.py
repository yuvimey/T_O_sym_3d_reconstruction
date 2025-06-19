import numpy as np
import math
import matplotlib.pyplot as plt


def estimate_relative_rotations(pf_norm, cache_file_name):
    """
    Estimate relative rotations between projections while considering shifts.
    """
    n_r, _, n_images = pf_norm.shape
    
    # Load cached data
    data = np.load(cache_file_name, allow_pickle=True)
    R = data['R']
    n_candidates = R.shape[2]
    S_self = np.zeros((n_images, n_candidates))
    est_rel_rots = np.zeros((n_images, n_images), dtype=int)
    
    # Prepare for common line computations
    n_pairs = math.comb(n_images, 2)
    ii_inds = np.zeros(n_pairs, dtype=int)
    jj_inds = np.zeros(n_pairs, dtype=int)
    clmats = [None] * n_pairs
    
    l_ij_ji_ind = data['l_ij_ji_ind']
    l_self_ind = data['l_self_ind']
    
    ind = 0
    for ii in range(n_images):
        for jj in range(ii + 1, n_images):
            ii_inds[ind] = ii
            jj_inds[ind] = jj
            ind += 1
    
    # Compute self common lines
    print("n_images: " + str(n_images))
    for p_i in range(n_images):
        print(f"     scls_correlation Iter ({p_i+1}/{n_images}) ")
        pf_norm_i = pf_norm[:, :, p_i]
        S_self[p_i, :] = scls_correlation(pf_norm_i, l_self_ind)
    
    # Compute common lines for pairs
    print("n_pairs: " + str(n_pairs))
    for ind in range(n_pairs):
        print(f"     max_correlation_pair_ind Iter ({ind+1}/{n_pairs}) ")
        p_i = ii_inds[ind]
        p_j = jj_inds[ind]
        clmats[ind] = max_correlation_pair_ind(pf_norm, p_i, p_j, S_self, n_candidates, l_ij_ji_ind)
    
    # Store rotation indices
    for ind in range(n_pairs):
        p_i = ii_inds[ind]
        p_j = jj_inds[ind]
        est_rel_rots[p_i, p_j], est_rel_rots[p_j, p_i] = np.unravel_index(clmats[ind], (n_candidates, n_candidates), order='F')
  
    return est_rel_rots

def max_correlation_pair_ind(pf_norm, p_i, p_j, S_self, n_candidates, l_ij_ji_ind):
    """
    Find the index of the maximal correlation between two images.
    """
    Corrs_cls = np.zeros_like(l_ij_ji_ind)
    
    pf_norm_i = pf_norm[:, :, p_i]
    pf_norm_j = pf_norm[:, :, p_j]
    
    Sij = np.outer(S_self[p_i, :], S_self[p_j, :])
    np.fill_diagonal(Sij, 0)

    Corrs = np.real(np.conj(pf_norm_i).T @ pf_norm_j)
    Corrs = np.ravel(Corrs, order='F')
    Corrs_cls = Corrs[l_ij_ji_ind]

    cl = np.prod(Corrs_cls, axis=1)
    c = cl.reshape((n_candidates, n_candidates), order='F')
    Sij *= c

    return np.argmax(Sij.ravel(order='F'))


def scls_correlation(pf_norm_i, l_self_ind):
    """
    Compute self common lines correlation.
    """
    Corrs_pi = np.real(np.conj(pf_norm_i).T @ pf_norm_i)
    Corrs_pi = np.ravel(Corrs_pi, order='F')
    Corrs_scls = Corrs_pi[l_self_ind]

    return np.prod(Corrs_scls, axis=1)