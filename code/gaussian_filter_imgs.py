import numpy as np

def gaussian_filter_imgs(npf):
    """
    Applies a Gaussian filter to the images.
    
    Parameters:
    - npf: numpy.ndarray
        A 3D array where each image npf[:,:,i] corresponds to the Fourier
        transform of projection i.
    
    Returns:
    - npf_out: numpy.ndarray
        The three-dimensional array where each slice is filtered.
    """
    n_r, n_theta, nImages = npf.shape
    
    # Create the full double-cover of rays
    pf = np.vstack([
        np.flip(npf[1:, n_theta // 2:, :], axis=0),
        npf[:, :n_theta // 2, :]
    ])
    #print("pf.shape 1: " + str(pf.shape))
    temp = pf
    pf = np.zeros((2 * n_r - 1, n_theta, nImages), dtype=np.complex128)
    pf[:, :n_theta // 2, :] = np.flip(temp, axis=0)
    pf[:, n_theta // 2:, :] = temp
    #print("pf.shape 2: " + str(pf.shape))
    
    pf = pf.reshape(2 * n_r - 1, n_theta * nImages, order='F')
    #print("pf.shape 3: " + str(pf.shape))
    rmax = n_r - 1
    rk = np.arange(-rmax, rmax + 1)
    H = np.sqrt(np.abs(rk)) * np.exp(-rk**2 / (2 * (rmax / 4)**2))
    pf *= H[:, np.newaxis]

    #print("H.shape: " + str(pf.shape))
    #print("pf.shape 4: " + str(pf.shape))
    
    pf = pf[:n_r, :]
    pf = np.flip(pf, axis=0)  # Flip back again by ascending frequency
    
    #print("pf.shape 5: " + str(pf.shape))
    npf_out = pf.reshape(n_r, n_theta, nImages, order='F')

    #print("npf_out.shape 6: " + str(npf_out.shape))
    return npf_out