import numpy as np
from nufft import nufft2

def cryo_pft(p, n_r, n_theta, precision='single'):
    """
    Compute the polar Fourier transform of projections with resolution n_r in
    the radial direction and resolution n_theta in the angular direction.

    Parameters:
    - p: 2D or 3D NumPy array (x, y, projections)
    - n_r: number of radial samples
    - n_theta: number of angular samples (must be even)
    - precision: 'single' or 'double' (currently not used)

    Returns:
    - pf: polar Fourier transform, shape (n_r, n_theta, n_proj)
    - freqs: (n_r * n_theta, 2) array of frequency sampling points (omega_x, omega_y)
    """

    n_proj = 1
    if p.ndim == 3:
        n_proj = p.shape[2]

    if n_theta % 2 != 0:
        raise ValueError("`n_theta` must be even.")

    omega0 = 2 * np.pi / (2 * n_r - 1)
    dtheta = 2 * np.pi / n_theta

    freqs = np.zeros((n_r * n_theta // 2, 2))
    for j in range(n_theta // 2):
        for k in range(n_r):
            freqs[j * n_r + k, 0] = (k) * omega0 * np.sin(j * dtheta)
            freqs[j * n_r + k, 1] = (k) * omega0 * np.cos(j * dtheta)

    if not (p.dtype == np.float64 or p.dtype == np.float32):
        raise TypeError("Images data type can be 'single' or 'double'")

    pf = nufft2(p, -freqs.T)
    pf = pf.reshape((n_r, n_theta // 2, n_proj), order='F')
    pf = np.concatenate((pf, np.conj(pf)), axis=1)

    return pf, freqs