import numpy as np

def check_imaging_params(params, L=None, n=None):
    """
    Check imaging parameters structure for consistency.
    
    Parameters:
        params (dict): Imaging parameters structure containing:
            - 'rot_matrices': A (3, 3, n) array of rotation matrices.
            - 'ctf': A (L, L, K) array of CTF images (centered Fourier transforms).
            - 'ctf_idx': A vector of length n mapping images to CTF indices.
            - 'ampl': A vector of length n specifying amplitude multipliers.
            - 'shifts': A (2, n) array of image shifts.
        L (int, optional): The size of the images. If None, inferred from `params['ctf']`.
        n (int, optional): The number of images. If None, inferred from `params['rot_matrices']`.
    
    Raises:
        ValueError: If any field is missing or has inconsistent dimensions.
    """
    required_fields = ['rot_matrices', 'ctf', 'ctf_idx', 'ampl', 'shifts']
    for field in required_fields:
        if field not in params:
            raise ValueError(f"Parameters must have `{field}` field.")
    
    if L is None:
        L = params['ctf'].shape[0]
    
    if n is None:
        n = params['rot_matrices'].shape[2]
    
    if params['rot_matrices'].shape != (3, 3, n):
        raise ValueError("Field `rot_matrices` must be of size (3, 3, n).")
    
    if params['ctf'].shape[0:2] != (L, L):
        print(f"params['ctf'].shape[0:2]: {params['ctf'].shape[0:2]} L={L}")
        raise ValueError("Field `ctf` must be of size (L, L, K).")
    
    if params['ctf_idx'].shape != (n,):
        raise ValueError("Field `ctf_idx` must be a vector of length n.")
    
    if params['ampl'].shape != (n,):
        raise ValueError("Field `ampl` must be a vector of length n.")
    
    if not np.all((params['ctf_idx'] >= 1) & (params['ctf_idx'] <= params['ctf'].shape[2])):
        raise ValueError("Field `ctf_idx` must have positive integer entries within valid range.")