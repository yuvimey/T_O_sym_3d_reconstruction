import numpy as np
from dirac_basis import dirac_basis
from cryo_kernel import cryo_mean_kernel_f, circularize_kernel_f
from im import im_backproject
from cryo_conj_grad_mean import cryo_conj_grad_mean
from check_imaging_params import check_imaging_params

out_dir = './TEST_T'

def cryo_estimate_mean(im, params, basis=None, mean_est_opt=None):
    """
    Estimate mean using least squares and conjugate gradients.

    Parameters:
        im (ndarray): L x L x n array containing projection images.
        params (dict): Imaging parameters structure containing:
            - 'rot_matrices': 3 x 3 x n array of projection rotation matrices.
            - 'ctf': L x L x K array of CTF images.
            - 'ctf_idx': n-length vector of CTF indices for each image.
            - 'ampl': n-length vector of amplitude multipliers for each image.
            - 'shifts': 2 x n array of x and y offsets applied after projection.
        basis (optional): Basis object for representing volumes (default: None).
        mean_est_opt (dict, optional): Options structure:
            - 'precision': Kernel precision, either 'double' (default) or 'single'.
            - 'half_pixel': Centers rotation around a half-pixel if True (default: False).
            - 'batch_size': Batch size for kernel computation (default: None, meaning no batches).
            - 'preconditioner': Preconditioner for conjugate gradient method:
                - 'none': No preconditioner.
                - 'circulant': Uses a circulant approximation as a preconditioner (default).
                - function handle: Custom preconditioner function.

    Returns:
        mean_est (ndarray): Estimated mean volume (L x L x L).
        cg_info (dict): Conjugate gradient method information.
    """

    if basis is None:
        basis = []

    if mean_est_opt is None:
        mean_est_opt = {}

    L = im.shape[0]
    n = im.shape[2]

    check_imaging_params(params, L, n)

    # Set default options
    mean_est_opt.setdefault('preconditioner', 'circulant')
    mean_est_opt.setdefault('precision', 'double')

    if not basis:
        basis = dirac_basis(np.full(3, L))

    mean_est_opt.setdefault('batch_size', 1)
    kernel_f = cryo_mean_kernel_f(L, params, mean_est_opt).squeeze()
    precond_kernel_f = None

    if isinstance(mean_est_opt['preconditioner'], str):
        if mean_est_opt['preconditioner'] == 'none':
            precond_kernel_f = None
        elif mean_est_opt['preconditioner'] == 'circulant':
            precond_kernel_f = circularize_kernel_f(kernel_f)

        else:
            raise ValueError("Invalid preconditioner type.")

        # Reset so this is not used by `conj_grad` function
        mean_est_opt['preconditioner'] = lambda x: x
    
    im_bp = cryo_mean_backproject(im, params, mean_est_opt)

    mean_est, cg_info = cryo_conj_grad_mean(kernel_f, im_bp, basis, precond_kernel_f, mean_est_opt)

    return mean_est, cg_info

def cryo_mean_backproject(im, params, mean_est_opt={}):
    """
    Backproject images for mean volume estimation.
    
    Parameters:
        im (ndarray): L-by-L-by-n array containing images to be backprojected.
        params (dict): Imaging parameters including rotation matrices, CTFs, amplitude multipliers, and shifts.
        mean_est_opt (dict, optional): Options for precision, half-pixel adjustment, and batch processing.
    
    Returns:
        im_bp (ndarray): The backprojected images, averaged over the whole dataset.
    """
    if im.shape[0] != im.shape[1] or im.shape[0] == 1 or im.ndim > 3:
        raise ValueError("Input `im` must be of size L-by-L-by-n for L > 1.")
    
    L = im.shape[0]
    n = im.shape[2]
    
    mean_est_opt.setdefault('precision', 'double')
    mean_est_opt.setdefault('half_pixel', False)
    mean_est_opt.setdefault('batch_size', None)
    
    if mean_est_opt['precision'] == 'single':
        im = im.astype(np.float32)
    
    im_bp = im_backproject(im, params['rot_matrices'], mean_est_opt['half_pixel'])
    
    return im_bp / n

