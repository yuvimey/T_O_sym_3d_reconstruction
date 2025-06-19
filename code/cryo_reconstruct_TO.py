import numpy as np
from dirac_basis import dirac_basis
from cryo_TO_group_elements import cryo_TO_group_elements
from cryo_estimate_mean import cryo_estimate_mean

def cryo_reconstruct_TO(symmetry, projs, rots):
    """
    Reconstructs a T or O symmetric molecule.
    
    Parameters:
        symmetry: str
            Symmetry type: 'T' or 'O'.
        projs: np.ndarray
            Array of projection images.
        rots: np.ndarray
            Array of estimated rotation matrices for each projection image.
        n_theta: int
            Angular resolution for common lines detection (default: 360).
        n_r: int
            Radial resolution for common line detection (default: 50% of image size).
    
    Returns:
        vol: np.ndarray
            Reconstructed volume.
        rotations: np.ndarray
            Rotation matrices used in reconstruction.
    """
    
    gR, _ = cryo_TO_group_elements(symmetry)
    n_gR = gR.shape[2]
    
    n_images = rots.shape[2]
    Rs_TO = np.zeros((3, 3, n_images * n_gR))
    for k in range(n_images):
        rot = rots[:, :, k]
        for s in range(n_gR):
            Rs_TO[:, :, s * n_images + k] = gR[:, :, s] @ rot
    
    n = projs.shape[0]
    projs_TO = np.tile(projs, (1, 1, n_gR))
    
    print('Reconstructing')
    params = {
        'rot_matrices': Rs_TO,
        'ctf': np.ones((n, n, projs_TO.shape[2])),
        'ctf_idx': np.ones(projs_TO.shape[2], dtype=int),
        'shifts': None,  #dx_TO.T,
        'ampl': np.ones(projs_TO.shape[2], dtype=int)
    }
    
    mean_est_opt = {
        'max_iter': 100,
        'rel_tolerance': 1.0e-6,
        'verbose': True,
        'precision': 'single'
    }
    
    basis = dirac_basis((n, n, n))
    
    v1, _  = cryo_estimate_mean(projs_TO.astype(np.float64), params, basis, mean_est_opt)
    
    vol = np.real(v1)
    rotations = Rs_TO
    
    return vol, rotations
