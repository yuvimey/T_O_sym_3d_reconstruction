import numpy as np
import os
from scipy.linalg import inv
from cryo_TO_create_cache import cryo_TO_create_cache
from cryo_raynormailze import cryo_raynormalize
from cryo_pft import cryo_pft
from handedness_synchronization_TO import handedness_synchronization_TO
from mask_fuzzy import mask_fuzzy
from cryo_reconstruct_TO import cryo_reconstruct_TO
from estimate_rotations_synchronization import estimate_rotations_synchronization
import mrcfile
from estimate_relative_rotations import estimate_relative_rotations

def cryo_abinitio_TO(sym, instack, outvol, cache_file_name=None,
                     n_theta=360, n_r_perc=50, mask_radius_perc=0):
    """
    Ab initio reconstruction of a T or O symmetric molecule.
    
    Parameters:
        sym: str
            'T' for tetrahedral symmetry, 'O' for octahedral symmetry.
        instack: str
            Name of MRC file containing the projections.
        outvol: str
            Name of MRC file to save the reconstructed volume.
        cache_file_name: str, optional
            The mat file name containing candidate rotation matrices and common lines indices.
        n_theta: int, optional
            Angular resolution for common lines detection (default 360).
        n_r_perc: int, optional
            Radial resolution as percentage of image size (default 50).
        max_shift_perc: int, optional
            Maximal 1D shift in pixels as percentage of image width (default 15).
        shift_step: float, optional
            Resolution of shift estimation in pixels (default 0.5).
        mask_radius_perc: int, optional
            Masking radius as percentage of image size (default 50).
        refq: array-like, optional
            A 4 x n_images array holding ground truth quaternions.
    """
    if sym not in ['T', 'O']:
        raise ValueError("First argument must be 'T' or 'O' (tetrahedral or octahedral symmetry)")
    
    folder_recon_mrc_fname = os.path.dirname(outvol)
    if folder_recon_mrc_fname and not os.path.exists(folder_recon_mrc_fname):
        raise ValueError(f'Folder {folder_recon_mrc_fname} does not exist. Please create it first.')
    
    if cache_file_name is None or not os.path.exists(cache_file_name):
        print('Cache file not supplied.')
        cache_dir_full_path = os.path.dirname(outvol) or os.getcwd()
        print(f'Creating cache file under folder: {cache_dir_full_path}')
        cache_file_name = cryo_TO_create_cache(sym)
    
    print(f'Loading MRC image stack file: {instack}')
    projs = mrcfile.read(instack)
    n_images = projs.shape[2]
    print(f'Projections loaded. Using {n_images} projections of size {projs.shape[0]} x {projs.shape[1]}')
    mask_radius = int(np.ceil(projs.shape[0] * mask_radius_perc / 100))

    if mask_radius > 0:
        print(f'Masking projections. Masking radius is {mask_radius} pixels')
        masked_projs, _ = mask_fuzzy(projs, mask_radius)
    else:
        masked_projs = projs
    masked_projs = projs
    
    n_r = int(np.ceil(projs.shape[0] * n_r_perc / 100))
    print('Computing the polar Fourier transform of projections')
    npf, _ = cryo_pft(masked_projs, n_r, n_theta, 'double')
    print('Gaussian filtering images')
    pf_norm = cryo_raynormalize(npf)
    
    print('Computing all relative rotations')
    est_rel_rots = estimate_relative_rotations(pf_norm, cache_file_name)
    
    print('Handedness synchronization')

    u_G = handedness_synchronization_TO(sym, est_rel_rots, cache_file_name)
        
    print('Estimating rotations')
    rots, _ = estimate_rotations_synchronization(est_rel_rots, u_G, cache_file_name)
    rots_t = np.transpose(rots, (1, 0, 2))
        
    print('Reconstructing ab initio volume')

    estimatedVol, _ = cryo_reconstruct_TO(sym, projs, rots)
    estimatedVol = np.squeeze(estimatedVol)
    mrcfile.write(outvol, estimatedVol.astype(np.float32), voxel_size=1, overwrite=True)