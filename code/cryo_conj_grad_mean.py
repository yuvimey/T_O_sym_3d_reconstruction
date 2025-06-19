import numpy as np
from conj_grad import conj_grad
from scipy.sparse.linalg import cg, LinearOperator

def cryo_conj_grad_mean(kernel_f, im_bp, basis, precond_kernel_f=None, mean_est_opt=None):
    """
    Solve for mean volume using conjugate gradient.
    
    Parameters:
        kernel_f (np.ndarray): The centered Fourier transform of the projection-backprojection operator.
        im_bp (np.ndarray): An array of size (L, L, L) containing the backprojected images.
        basis: A basis object used for representing volumes.
        precond_kernel_f (np.ndarray, optional): Fourier transform of a preconditioning kernel (default None).
        mean_est_opt (dict, optional): Options passed to the conjugate gradient solver (default None).
    
    Returns:
        tuple: (mean_est, cg_info), where mean_est is the least-squares estimate and cg_info contains CG solver details.
    """
    if mean_est_opt is None:
        mean_est_opt = {}
    
    L = im_bp.shape[0]
    
    if im_bp.ndim != 3 or im_bp.shape != (L, L, L):
        raise ValueError("Input 'im_bp' must be an array of size (L, L, L).")
    
    if not 'evaluate' in basis or not 'evaluate_t' in basis:
        raise ValueError("Input 'basis' must be a basis object representing volumes of size (L, L, L).")
    
    def fun(vol_basis):
        return apply_mean_kernel(vol_basis, kernel_f, basis, mean_est_opt)
    
    if precond_kernel_f is not None:
        def precond_fun(vol_basis):
            return apply_mean_kernel(vol_basis, precond_kernel_f, basis, mean_est_opt)
        mean_est_opt['preconditioner'] = lambda x: precond_fun(x)
    
    im_bp_basis = basis['evaluate_t'](im_bp)
    
    mean_est_basis, _, info = conj_grad(fun, im_bp_basis, mean_est_opt)
    
    mean_est = basis['evaluate'](mean_est_basis)
    
    return mean_est, info

def apply_mean_kernel(vol_basis, kernel_f, basis, mean_est_opt):
    """
    Applies the mean kernel represented by convolution.
    
    Parameters:
        vol_basis: The volume to be convolved, stored in the basis coordinates.
        kernel_f: The centered Fourier transform of the convolution kernel.
        basis: A basis object corresponding to the basis used to store `vol_basis`.
        mean_est_opt: Options structure (currently unused).
    
    Returns:
        np.ndarray: The convolved volume in basis coordinates.
    """
    vol = basis['evaluate'](vol_basis)
    vol = np.squeeze(vol)
    vol = cryo_conv_vol(vol, kernel_f)
    vol = vol[:,:,:,np.newaxis]
    return basis['evaluate_t'](vol)

def cryo_conv_vol(x, kernel_f):
    """
    Convolve volume(s) with a kernel in Fourier space.

    Parameters:
        x (ndarray): An array of shape (N, N, N, ...) representing one or more cubic volumes.
        kernel_f (ndarray): A Fourier-transformed cubic convolution kernel (centered).
                            Must be larger than x in the first three dimensions.

    Returns:
        ndarray: Convolved volumes with the same shape as the input.
    """
    N = x.shape[0]

    # Flatten/prepare extra dimensions (batching or other dims beyond the 3D volume)

    if not (x.shape[0] == x.shape[1] == x.shape[2] == N):
        raise ValueError("Volumes in `x` must be cubic.")

    is_singleton = (x.ndim == 3)

    N_ker = kernel_f.shape[0]

    if kernel_f.shape != (N_ker, N_ker, N_ker):
        raise ValueError("Convolution kernel `kernel_f` must be cubic.")

    # Shift the kernel to match centered Fourier transform
    kernel_f = np.fft.ifftshift(np.fft.ifftshift(np.fft.ifftshift(kernel_f, axes=0), axes=1), axes=2)

    if is_singleton:
        x = np.fft.fftn(x, s=[N_ker]*3)
    else:
        x = np.fft.fft(x, n=N_ker, axis=0)
        x = np.fft.fft(x, n=N_ker, axis=1)
        x = np.fft.fft(x, n=N_ker, axis=2)
    
    # Apply convolution in Fourier domain
    x = x * kernel_f[..., np.newaxis] if not is_singleton else x * kernel_f

    if is_singleton:
        x = np.fft.ifftn(x)
        x = x[:N, :N, :N]
    else:
        x = np.fft.ifft(x, axis=0)
        x = x[:N, :, :, :]
        x = np.fft.ifft(x, axis=1)
        x = x[:, :N, :, :]
        x = np.fft.ifft(x, axis=2)
        x = x[:, :, :N, :]

    x = np.real(x)

    return x