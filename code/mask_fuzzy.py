import numpy as np
from scipy import special

def mask_fuzzy(projections, rad):
    """
    Mask the projections with a radius mask of size 'rad', and compute the
    standard deviation of noise 'sigma'.
    
    Parameters:
    - projections: numpy.ndarray
        3D array of projections.
    - rad: int
        Radius of the fuzzy mask.
    
    Returns:
    - projections: numpy.ndarray
        Masked projections.
    - sigma: float
        Standard deviation of noise.
    """
    siz = projections.shape[0]
    center = (siz + 1) / 2
    m = fuzzymask(siz, 2, rad, 2, [center, center])
    
    I, J = np.meshgrid(np.arange(1, siz + 1), np.arange(1, siz + 1), indexing='ij')
    r = np.sqrt((I - center) ** 2 + (J - center) ** 2)
    ind = r > rad
    
    n_proj = projections.shape[2]
    projections = projections.reshape(siz ** 2, n_proj, order='F')
    noise = projections[ind.flatten(order='F'), :]
    sigma = np.std(noise)
    
    # Apply fuzzy mask
    projections *= m.flatten(order='F')[:, np.newaxis]
    projections = projections.reshape(siz, siz, n_proj, order='F')
    
    return projections, sigma

def fuzzymask(n, dims, r0, risetime, origin=None):
    """
    Create a centered 1D, 2D, or 3D disc of radius r0, using an error function
    with effective risetime.
    
    Parameters:
    - n: int
        Size of the array (assumed square/cubic if 2D or 3D).
    - dims: int
        Number of dimensions (1, 2, or 3).
    - r0: float
        Radius of the mask.
    - risetime: float
        Transition sharpness.
    - origin: list or numpy.ndarray, optional
        Center coordinates. Default is n/2 + 1 for each dimension.
    
    Returns:
    - m: numpy.ndarray
        Generated mask.
    """
    ctr = n / 2 + 1 if origin is None else np.array(origin)
    k = 1.782 / risetime
    
    if dims == 1:
        r = np.abs(np.arange(1, n + 1) - ctr)
    elif dims == 2:
        x, y = np.meshgrid(np.arange(1, n + 1) - ctr[0], np.arange(1, n + 1) - ctr[1], indexing='ij')
        r = np.sqrt(x**2 + y**2)
    elif dims == 3:
        x, y, z = np.meshgrid(
            np.arange(1, n + 1) - ctr[0],
            np.arange(1, n + 1) - ctr[1],
            np.arange(1, n + 1) - ctr[2], indexing='ij'
        )
        r = np.sqrt(x**2 + y**2 + z**2)
    else:
        raise ValueError(f"Wrong number of dimensions: {dims}")
    
    return 0.5 * (1 - special.erf(k * (r - r0)))