import numpy as np
import math
from cryo_TO_group_elements import cryo_TO_group_elements
from uppertri_ijtoind_vec import uppertri_ijtoind_vec
from signs_power_method import signs_power_method
from lin2sub3_map import lin2sub3_map
from multi_Jify import multi_Jify
from multiprod import multiprod

def handedness_synchronization_TO(symmetry, est_rel_rots, cache_file_name):
    n_images = est_rel_rots.shape[0]
    gR, _ = cryo_TO_group_elements(symmetry)
    data = np.load(cache_file_name, allow_pickle=True)
    R = data['R']
    
    n_gR = gR.shape[2]
    n_pairs = n_images * (n_images - 1) // 2
    Rijs = np.zeros((3, 3, n_pairs, n_gR))
    
    for i in range(n_images - 1):
        for j in range(i + 1, n_images):
            pair_ind = uppertri_ijtoind_vec(i, j, n_images)
            for k in range(n_gR):
                Rijs[:, :, pair_ind, k] = R[:, :, est_rel_rots[i, j]] @ gR[:, :, k] @ R[:, :, est_rel_rots[j, i]].T
    
    return jsync_TO(n_gR, Rijs, n_images)

def jsync_TO(n_gR, Rijs, K):
    print('Syncing relative rotations')
    J_list = outer_sync_brute_eff(n_gR, Rijs, K)
    print('Done matching triplets, now running power method...')
    u_G, _, evals, _, _ = signs_power_method(K, J_list.astype(float), 3, 0)
    print(f'top 3 eigenvalues = {evals[0]:.2f} {evals[1]:.2f} {evals[2]:.2f}')
    print('Done syncing handedness')
    return u_G

def outer_sync_brute_eff(n_gR, Rijs, K):
    nworkers = 2 # 32
    J_list = np.zeros(math.comb(K, 3))
    ntrip = math.comb(K, 3)
    ntrip_per_worker = ntrip // nworkers
    ntrip_last_worker = ntrip - ntrip_per_worker * (nworkers - 1)
    iter_lin_idx = np.zeros(nworkers + 1, dtype=int)
    
    for i in range(1, nworkers):
        iter_lin_idx[i] = iter_lin_idx[i - 1] + ntrip_per_worker
    iter_lin_idx[nworkers] = iter_lin_idx[nworkers - 1] + ntrip_last_worker
    
    workers_res = []
    for i in range(nworkers):
        lin_idx_loc = iter_lin_idx
        workers_res.append(outer_sync_brute_i(n_gR, Rijs, K, lin_idx_loc[i], lin_idx_loc[i + 1]))
    
    for i in range(nworkers):
        J_list[iter_lin_idx[i]:iter_lin_idx[i + 1]] = workers_res[i]
    
    return J_list

def outer_sync_brute_i(n_gR, Rijs, K, from_idx, to_idx):
    J_list_i = np.zeros(math.comb(K, 3))
    norms = np.zeros(math.comb(K, 3))
    
    ntrip = to_idx - from_idx
    trip_idx = lin2sub3_map(K)[from_idx:to_idx]
    
    ks = [uppertri_ijtoind_vec(trip_idx[:, 0], trip_idx[:, 1], K)]
    ks += [uppertri_ijtoind_vec(trip_idx[:, 0], trip_idx[:, 2], K)]
    ks += [uppertri_ijtoind_vec(trip_idx[:, 1], trip_idx[:, 2], K)]
    Rijs_t = np.transpose(Rijs, (1, 0, 2, 3))
    for t in range(ntrip):
        k1 = ks[0][t]
        k2 = ks[1][t]
        k3 = ks[2][t]
        Rij = Rijs[:, :, k1, :]
        Rijk = np.asarray([Rijs[:, :, k2, :], Rijs_t[:, :, k3, :]]).transpose((1, 2, 0, 3))
        
        final_votes = np.zeros(4)
        final_votes[0], prod_arr = compare_rot_brute_eff(n_gR, Rij, Rijk)
        final_votes[1], _ = compare_rot_brute_eff(n_gR, Rij, None, multi_Jify(prod_arr))
        k2_Jified = multi_Jify(Rijk[:, :, 1, :])
        Rijk[:, :, 1, :] = k2_Jified
        final_votes[2], prod_arr = compare_rot_brute_eff(n_gR, Rij, Rijk)
        final_votes[3], _ = compare_rot_brute_eff(n_gR, Rij, None, multi_Jify(prod_arr))
        
        norms[from_idx + t], decision = min(final_votes), np.argmin(final_votes)
        J_list_i[from_idx + t] = decision
    
    return J_list_i[from_idx:to_idx]

def compare_rot_brute_eff(n_gR, Rij, Rijk, Jified_rot=None):
    if Jified_rot is None:
        prod_arr = multiprod(Rijk[:, :, 0, :], Rijk[:, :, 1, :])
        prod_arr = np.transpose(prod_arr,(1,2,0))
    else:
        prod_arr = Jified_rot
    
    arr = np.zeros((3, 3, n_gR, n_gR, n_gR))
    prod_arr_reshaped = np.array([prod_arr]).transpose(1, 2, 0, 3)
    for i in range(n_gR):
        Rij_tiled = np.tile(Rij[:, :, i, np.newaxis, np.newaxis], (1, 1, n_gR, n_gR))
        arr[:, :, :, :, i] = prod_arr_reshaped - Rij_tiled
    
    arr = arr.reshape(9, n_gR * n_gR * n_gR, order='F')
    arr = np.sum(arr ** 2, axis=0)
    vote = np.sum(np.sort(arr)[:n_gR * n_gR])
    
    return vote, prod_arr
