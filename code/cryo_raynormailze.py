import numpy as np

def cryo_raynormalize(pf):
    """
    Normalize a dataset of Fourier rays so that each ray has energy 1.
    
    Parameters:
    - pf: numpy.ndarray
        3D array of the Fourier transform of the projections.
        pf[:,:,k] is the polar Fourier transform of the k-th projection.
    
    Returns:
    - pf2: numpy.ndarray
        Normalized Fourier transform array.
    """
    pf2 = np.copy(pf)  # Create a copy for normalization
    n_proj = pf.shape[2] if pf.ndim == 3 else 1
    n_theta = pf.shape[1]
    
    for k in range(n_proj):
        for j in range(n_theta):
            nr = np.linalg.norm(pf2[:, j, k])
            if nr < 1.0e-13:
                print(f'Warning: Ray norm is close to zero. k={k}, j={j}')
            pf2[:, j, k] /= nr
    
    return pf2