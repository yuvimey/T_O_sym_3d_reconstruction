import numpy as np
from rotated_grids import rotated_grids
from nufft import anufft3
from check_imaging_params import check_imaging_params
import numpy as np

def circularize_kernel_f(kernel_f):
    dims = kernel_f.ndim

    #print("kernel_f: " + str(kernel_f))
    #print("kernel_f.shape: " + str(kernel_f.shape))

    kernel_f = mdim_ifftshift(kernel_f, axes=range(dims))
    kernel = np.fft.ifftn(kernel_f)
    kernel = mdim_fftshift(kernel, axes=range(dims))

    kernel_circ = kernel
    #print("kernel_circ 0: " + str(kernel_circ))
    #print("kernel_circ.shape 0: " + str(kernel_circ.shape))
    for dim in range(dims):
        kernel_circ = circularize_kernel_1d(kernel_circ, dim)
    
    kernel_circ = mdim_ifftshift(kernel_circ, axes=range(dims))

    kernel_circ_f = np.fft.fftn(kernel_circ)
    kernel_circ_f = mdim_fftshift(kernel_circ_f, axes=range(dims))

    return kernel_circ_f

def circularize_kernel_1d(kernel, dim):
    sz = kernel.shape
    N = sz[dim] // 2

    # Prepare slicing to extract "top" half.
    slices_top = [slice(None)] * kernel.ndim
    slices_top[dim] = slice(0, N)

    # Multiplier for weighted average (top half)
    shape_mult = [1]*dim + [N] + [1]*(kernel.ndim - dim - 1)
    mult_top = np.reshape(np.arange(N) / N, shape_mult, order='F')

    kernel_circ = mult_top * kernel[tuple(slices_top)]

    # Prepare slicing to extract "bottom" half.
    slices_bot = [slice(None)] * kernel.ndim
    slices_bot[dim] = slice(N, 2*N)

    # Multiplier for weighted average (bottom half)
    mult_bot = np.reshape(np.arange(N, 0, -1) / N, shape_mult, order='F')

    kernel_circ += mult_bot * kernel[tuple(slices_bot)]

    # Shift to ensure zero frequency is at the center.
    kernel_circ = np.fft.fftshift(kernel_circ, axes=dim)

    return kernel_circ

def mdim_fftshift(arr, axes):
    for ax in axes:
        arr = np.fft.fftshift(arr, axes=ax)
    return arr

def mdim_ifftshift(arr, axes):
    for ax in axes:
        arr = np.fft.ifftshift(arr, axes=ax)
    return arr

def cryo_mean_kernel_f(L, params, mean_est_opt=None):
    if mean_est_opt is None:
        mean_est_opt = {}

    check_imaging_params(params, L, None)

    n = params['rot_matrices'].shape[2]

    # Fill missing options with defaults
    mean_est_opt.setdefault('precision', 'double')
    mean_est_opt.setdefault('half_pixel', False)
    
    pts_rot = rotated_grids(L, params['rot_matrices'], mean_est_opt['half_pixel'])
    shape = tuple([2 * L] * 3)
    mean_kernel_f = np.zeros(shape, dtype=np.complex128)#mean_est_opt['precision'])

    ctf = params['ctf']
    ctf_idx = params['ctf_idx']
    ampl = params['ampl']

    filter_ = np.abs(ctf[:, :, ctf_idx]) ** 2
    filter_ = filter_.astype(mean_est_opt['precision'])
    filter_ *= (ampl[np.newaxis, :1] ** 2)

    if L % 2 == 0 and not mean_est_opt['half_pixel']:
        pts_rot = pts_rot[:, 1:, 1:,:]
        filter_ = filter_[1:, 1:,:]

    filter_ = im_to_vec(filter_)

    pts_rot = pts_rot.reshape(3, -1, order='F')
    filter_ = filter_.reshape(-1, 1, order='F')

    mean_kernel =  (1/n)*((1 / L) ** 2) * anufft3(filter_, pts_rot, (2 * L, 2 * L, 2 * L)).squeeze()

    # Ensure symmetry
    mean_kernel[0, :, :] = 0
    mean_kernel[:, 0, :] = 0
    mean_kernel[:, :, 0] = 0

    # Take Fourier transform
    mean_kernel = np.fft.ifftshift(np.fft.ifftshift(np.fft.ifftshift(mean_kernel, 0), 1), 2)
    mean_kernel_f = np.fft.fftn(mean_kernel)
    mean_kernel_f = np.fft.fftshift(np.fft.fftshift(np.fft.fftshift(mean_kernel_f, 0), 1), 2)
    if np.isnan(mean_kernel_f).any():
        print("volume has Nan!")
    if np.isinf(mean_kernel_f).any():
        print("volume has inf!")

    return np.real(mean_kernel_f)

def im_to_vec(im):
    N = im.shape[0]

    if im.shape[1] != N:
        raise ValueError("Images in `im` must be square.")

    sz = im.shape
    vec = im.reshape((N * N, *sz[2:], 1), order='F')
    return vec