import numpy as np
import finufft

def nufft2(im, fourier_pts, nufft_opt=None):
    """
    NUFFT2 Wrapper for non-uniform FFT (2D)

    Parameters:
    - im: An (N1 x N2) or (N1 x N2 x L) NumPy array representing image(s).
    - fourier_pts: 2 x K array of Fourier space points in the range [-pi, pi].
    - nufft_opt: Dictionary with options. Supports:
        - 'epsilon': Desired precision (default 1e-15).

    Returns:
    - im_f: The Fourier transform of im at the frequencies specified in fourier_pts.
    """
    if nufft_opt is None:
        nufft_opt = {}

    epsilon = nufft_opt.get('epsilon', 1e-15)

    if im.ndim < 2:
        raise ValueError("Input 'im' must be at least 2D.")

    if fourier_pts.ndim != 2 or fourier_pts.shape[0] != 2:
        raise ValueError("Input 'fourier_pts' must be of shape (2, K).")

    N1, N2 = im.shape[0], im.shape[1]
    K = fourier_pts.shape[1]

    # Normalize frequencies for finufft (range [-0.5, 0.5])
    kx = np.ascontiguousarray(fourier_pts[0, :]) / (2 * np.pi)
    ky = np.ascontiguousarray(fourier_pts[1, :]) / (2 * np.pi)

    im = np.ascontiguousarray(im)

    if im.ndim == 2:
        im_f = finufft.nufft2d2(kx, ky, im.astype(np.complex128), eps=epsilon)
    else:
        L = im.shape[2]
        im_f = np.zeros((K, L), dtype=np.complex128)
        for l in range(L):
            im_f[:, l] = finufft.nufft2d2(kx, ky, im[:, :, l].astype(np.complex128), eps=epsilon)
    return im_f


def nufft3(vol, fourier_pts, nufft_opt=None):
    """
    NUFFT3 Wrapper for non-uniform FFT (3D)

    Parameters:
    - vol: A 3D or 4D NumPy array (N1 x N2 x N3 [x L]) representing volume(s).
    - fourier_pts: 3 x K array of Fourier space points in the range [-pi, pi].
    - nufft_opt: Dictionary with options. Supports:
        - 'epsilon': Desired precision (default 1e-15).

    Returns:
    - vol_f: The Fourier transform of vol at the frequencies specified in fourier_pts.
    """
    if nufft_opt is None:
        nufft_opt = {}

    epsilon = nufft_opt.get('epsilon', 1e-15)

    if vol.ndim < 3:
        raise ValueError("Input 'vol' must be at least 3D.")

    if fourier_pts.ndim != 2 or fourier_pts.shape[0] != 3:
        raise ValueError("Input 'fourier_pts' must be of shape (3, K).")

    N1, N2, N3 = vol.shape[0], vol.shape[1], vol.shape[2]
    K = fourier_pts.shape[1]

    kx = np.ascontiguousarray(fourier_pts[0, :])
    ky = np.ascontiguousarray(fourier_pts[1, :])
    kz = np.ascontiguousarray(fourier_pts[2, :])

    if vol.ndim == 3:
        vol_f = finufft.nufft3d2(kx, ky, kz, vol.astype(np.complex128), eps=epsilon)
    else:
        L = vol.shape[3]
        vol_f = np.zeros((K, L), dtype=np.complex128)
        for l in range(L):
            vol_f[:, l] = finufft.nufft3d2(kx, ky, kz, vol[:, :, :, l].astype(np.complex128), eps=epsilon)

    return vol_f


def anufft3(vol_f, fourier_pts, sz, nufft_opt=None):
    """
    ANUFFT3 Wrapper for adjoint non-uniform FFT (3D)

    Parameters:
    - vol_f: Array of shape (K,) or (K, L) representing Fourier samples.
    - fourier_pts: 3 x K array of Fourier space points in range [-pi, pi].
    - sz: Tuple or list (N1, N2, N3) representing the output volume size.
    - nufft_opt: Dictionary with options. Supports:
        - 'epsilon': Desired precision (default 1e-15).

    Returns:
    - vol: The adjoint NUFFT result in spatial domain of shape sz (or sz + [L]).
    """
    if nufft_opt is None:
        nufft_opt = {}

    epsilon = nufft_opt.get('epsilon', 1e-15)

    if vol_f.ndim < 1:
        raise ValueError("Input 'vol_f' must be at least 1D (K,) or 2D (K, L).")

    K = vol_f.shape[0]

    if fourier_pts.ndim != 2 or fourier_pts.shape != (3, K):
        raise ValueError("Input 'fourier_pts' must be of shape (3, K).")

    if len(sz) != 3 or not all(isinstance(s, int) and s > 0 for s in sz):
        raise ValueError("Input 'sz' must be a tuple/list of 3 positive integers.")

    N1, N2, N3 = sz
    kx = np.ascontiguousarray(fourier_pts[0, :])
    ky = np.ascontiguousarray(fourier_pts[1, :])
    kz = np.ascontiguousarray(fourier_pts[2, :])

    if vol_f.ndim == 1:
        vol = finufft.nufft3d1(kx, ky, kz, vol_f.astype(np.complex128), (N1, N2, N3), isign=1, eps=epsilon)
    else:
        L = vol_f.shape[1]
        vol = np.zeros((N1, N2, N3, L), dtype=np.complex128)
        for l in range(L):
            vol[:, :, :, l] = finufft.nufft3d1(kx, ky, kz, vol_f[:, l].astype(np.complex128), (N1, N2, N3), isign=1, eps=epsilon)

    return vol