import numpy as np
from scipy.special import comb

def signs_power_method(N, J_list, n_eigs, verbose=0, measure_time=False, s=None):
    """
    Compute the first eigenvalues & eigenvectors of the Signs matrix using the power method (D2 version).
    
    Parameters:
    - N: int
        Number of projections on which Rij are based.
    - J_list: numpy.ndarray
        Array of relative rotations Rij before J-synchronization.
    - n_eigs: int
        Number of eigenvalues to compute.
    - verbose: int, optional (default: 0)
        0 for nothing, 1 for global printing, 2 for iterative printing.
    - measure_time: bool, optional (default: False)
        Whether to measure execution time.
    - s: int, optional
        Seed for random initialization.
    
    Returns:
    - J_sync: numpy.ndarray
        Vector of J-synchronization results.
    - J_significance: numpy.ndarray
        Vector expressing significance of each J-synchronization.
    - eigenvalues: numpy.ndarray
        Computed eigenvalues of the signs matrix.
    - itr: int
        Number of iterations performed.
    - dd: float
        Size of last change in the iterative estimated eigenvector.
    """
    if n_eigs <= 0:
        raise ValueError("n_eigs must be a positive integer!")
    
    if s is not None:
        np.random.seed(s)
    
    epsilon = 0.0000
    tol = 1e-4
    MAX_ITERATIONS = 100
    N_pairs = int(comb(N, 2))
    
    if verbose >= 2:
        print(f'Power method for signs matrix started, accuracy goal {epsilon}, max {MAX_ITERATIONS} iterations.')
    
    vec = np.random.rand(N_pairs, n_eigs)
    vec /= np.linalg.norm(vec)
    
    dd = 1
    eig_diff = np.inf
    itr = 0
    eigenvalues = np.zeros((3, 3))
    prev_eval = np.inf
    
    while itr < MAX_ITERATIONS and (abs(eigenvalues[0, 0] / (2 * N - 4)) > 1 + epsilon or abs(eigenvalues[0, 0] / (2 * N - 4)) < 1 - epsilon) and eig_diff > tol:
        itr += 1
        vec_new = signs_times_v_mex2(N, J_list, vec, n_eigs)
        vec_new = vec_new.reshape(vec.shape, order='F')
        vec_new, eigenvalues = np.linalg.qr(vec_new)
        
        dd = np.linalg.norm(vec_new[:, 0] - vec[:, 0])
        vec = vec_new
        eig_diff = abs(prev_eval - eigenvalues[0, 0])
        prev_eval = eigenvalues[0, 0]
        
        if verbose >= 2 :
            print(f'Iteration {itr:02d}: ||curr_evec-last_evec|| = {dd:.3f}')
    
    print(f'num of Jsync iterations: {itr}')
    eigenvalues = np.diag(eigenvalues)
    J_significance = np.abs(vec[:, 0])
    J_sync = np.sign(vec[:, 0])
    
    if verbose >= 1:
        print(f'Outer J Synchronization:\n\titerations needed: {itr}')
        print(f'\tsigns matrix eigenvalues (in practice vs. theoretical):\n\t\t{eigenvalues[0]:.0f}\t({2 * N - 4:.0f})')
        for i in range(1, min(n_eigs, 5)):
            print(f'\t\t{eigenvalues[i]:.0f}\t({N - 4:.0f})')
    
    return J_sync, J_significance, eigenvalues, itr, dd

def signs_times_v_mex2(N, best_vec, v, N_eigs):
    """
    Compute the multiplication of the signs matrix by the eigenvector candidate.
    
    Parameters:
    - N: int
        Number of projections.
    - best_vec: numpy.ndarray
        Array indicating best sign configurations.
    - v: numpy.ndarray
        Current eigenvector candidates (N-choose-2 x N_eigs).
    - N_eigs: int
        Number of eigenvectors computed.
    
    Returns:
    - v2: numpy.ndarray
        Result of the signs matrix times v multiplication (N-choose-2 x N_eigs).
    """
    N_pairs = int(comb(N, 2))
    v2 = np.zeros((N_pairs, N_eigs))
    
    def pair_idx(N, i, j):
        return (2 * N - i - 1) * i // 2 + j - i - 1
    
    def trip_idx(N, i, j, k):
        return (N * (N - 1) * (N - 2)) // 6 - ((N - i - 3) * (N - i - 2) * (N - i - 1)) // 6 - ((N - j - 1) * (N - j)) // 2 + k - j - 1
    
    signs_confs = np.ones((4, 3), dtype=int)
    signs_confs[1, [0, 2]] = -1
    signs_confs[2, [0, 1]] = -1
    signs_confs[3, [1, 2]] = -1
    
    for i in range(N - 2):
        for j in range(i + 1, N - 1):
            ij = pair_idx(N, i, j)
            jk = pair_idx(N, j, j)
            ik = ij
            
            for k in range(j + 1, N):
                jk += 1
                ik += 1
                ijk = trip_idx(N, i, j, k)
                best_i = int(best_vec[ijk])
                
                s_ij_jk = signs_confs[best_i, 0]
                s_ik_jk = signs_confs[best_i, 1]
                s_ij_ik = signs_confs[best_i, 2]
                
                for eig in range(N_eigs):
                    v2[ij, eig] += s_ij_jk * v[jk, eig] + s_ij_ik * v[ik, eig]
                    v2[jk, eig] += s_ij_jk * v[ij, eig] + s_ik_jk * v[ik, eig]
                    v2[ik, eig] += s_ij_ik * v[ij, eig] + s_ik_jk * v[jk, eig]
    
    return v2
