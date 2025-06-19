import numpy as np
from nufft import anufft3
from rotated_grids import rotated_grids

def im_translate(im, shifts):
    """
    Translate image by specified shifts using Fourier shift theorem.
    
    Parameters:
        im (np.ndarray): An array of shape (L, L, n) containing images to be translated.
        shifts (np.ndarray): An array of shape (2, n) specifying the shifts in pixels.
                             If it is a column vector of length 2, the same shifts are applied to each image.
    
    Returns:
        np.ndarray: The images translated by the shifts, with periodic boundaries.
    """
    n_im = im.shape[2]
    n_shifts = shifts.shape[1]
    
    if shifts.shape[0] != 2:
        raise ValueError("Input 'shifts' must be of size 2-by-n.")
    
    if n_shifts != 1 and n_im != n_shifts:
        raise ValueError("The number of shifts must be 1 or match the number of images.")
    
    if im.shape[0] != im.shape[1]:
        raise ValueError("Images must be square.")
    
    L = im.shape[0]
    grid = np.fft.ifftshift(np.arange(-L//2, L//2))
    om_x, om_y = np.meshgrid(grid, grid, indexing='ij')
    
    phase_shifts = (om_x[..., None] * shifts[0, :] / L + 
                    om_y[..., None] * shifts[1, :] / L)
    
    mult_f = np.exp(-2j * np.pi * phase_shifts)
    im_f = np.fft.fft2(im, axes=(0, 1))
    
    im_translated_f = im_f * mult_f
    im_translated = np.real(np.fft.ifft2(im_translated_f, axes=(0, 1)))
    
    return im_translated

def im_backproject(im, rot_matrices, half_pixel=False):
    """
    Backproject images along rotation using an adjoint NUFFT.
    
    Parameters:
        im (np.ndarray): An array of shape (L, L, n) containing images to backproject.
        rot_matrices (np.ndarray): A (3, 3, n) array of rotation matrices corresponding to viewing directions.
        half_pixel (bool, optional): If True, centers the rotation around a half-pixel (default False).
    
    Returns:
        np.ndarray: A (L, L, L, n) array of volumes corresponding to the backprojections.
    """
    L = im.shape[0]
    n = im.shape[2]
    
    if im.shape[1] != L:
        raise ValueError("im must be of the form L-by-L-by-K")
    
    if rot_matrices.shape[2] != im.shape[2]:
        raise ValueError("The number of rotation matrices must match the number of images.")
    pts_rot = rotated_grids(L, rot_matrices, half_pixel)
    pts_rot = pts_rot.reshape((3, (L**2) * n), order='F')

    im = im.transpose((1, 0, 2))
    
    im_f = (1/L ** 2) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(im, axes=(0,1)), axes=(0,1)),axes=(0,1))

    if L % 2 == 0:
        if not half_pixel:
            im_f[0, :, :] = 0
            im_f[:, 0, :] = 0

    im_f = im_f.reshape((L**2 * n), order='F')
    
    vol = anufft3(im_f, pts_rot, (L, L, L))
    
    return np.real(vol)

def im_filter(im, filter_f):
    """
    Filter image by a transfer function.
    
    Parameters:
        im (np.ndarray): An array of shape (L, L, n) containing images to be filtered.
        filter_f (np.ndarray): The centered Fourier transform of a filter (its transfer function) or a set of filters.
                               The first two dimensions of this filter must be equal to the first two dimensions of `im`.
                               If one filter is given, it is applied to all images.
                               If multiple filters are given, the shape must be compatible with `im`.
    
    Returns:
        np.ndarray: The filtered images.
    """
    n_im = im.shape[2]
    n_filter = filter_f.shape[2] if filter_f.ndim == 3 else 1
    
    if n_filter != 1 and n_filter != n_im:
        raise ValueError("The number of filters must either be 1 or match the number of images.")
    
    if im.shape[:2] != filter_f.shape[:2]:
        raise ValueError("The size of the images and filters must match.")
    
    im_f = np.fft.fft2(im, axes=(0, 1))
    im_filtered_f = im_f * filter_f
    im_filtered = np.real(np.fft.ifft2(im_filtered_f, axes=(0, 1)))
    
    return im_filtered