import numpy as np

def cryo_addnoise(projections, SNR, noise_type='gaussian', seed=None):
    """
    Add additive noise to projection images.
    
    Parameters:
        projections (np.ndarray): 3D stack of projections, where projections[:, :, k] is the k'th projection.
        SNR (float): Signal-to-noise ratio of the output noisy projections.
        noise_type (str, optional): 'color' or 'gaussian' noise type (default: 'gaussian').
        seed (int, optional): Seed for random number generator to ensure reproducibility (default: None).
    
    Returns:
        tuple: (noisy_projections, noise, I, sigma)
        - noisy_projections (np.ndarray): Stack of noisy projections.
        - noise (np.ndarray): Stack containing the additive noise.
        - I (np.ndarray): Normalized power spectrum of the noise.
        - sigma (float): Standard deviation of Gaussian noise for the required SNR.
    """
    p, _, K = projections.shape
    noisy_projections = np.zeros_like(projections)
    noise = np.zeros_like(projections)
    
    sigma = np.sqrt(np.var(projections[:, :, 0].reshape(-1, order='F')) / SNR)
    
    # Initialize random number generator
    if seed is not None:
        np.random.seed(seed)
    
    # Define index range for noise cropping
    lowidx = -(p - 1) // 2 + p if p % 2 == 1 else -p // 2 + p
    highidx = (p - 1) // 2 + p if p % 2 == 1 else p // 2 + p - 1
    
    # Compute color noise response
    I = cart2rad(2 * p + 1)
    I = 1 / np.sqrt(1 + I**2)
    I /= np.linalg.norm(I)
    noise_response = np.sqrt(I)
    
    for k in range(K):
        gn = np.random.randn(2 * p + 1, 2 * p + 1)
        
        if noise_type.lower() == 'gaussian':
            cn = gn
        else:
            cn = np.real(np.fft.ifft2(np.fft.fft2(gn, axes=(0,1)) * noise_response))
        
        cn = cn[lowidx:highidx+1, lowidx:highidx+1]
        cn = cn / np.std(cn)
        cn = cn * sigma
        
        noisy_projections[:, :, k] = projections[:, :, k] + cn
        noise[:, :, k] = cn
    
    return noisy_projections, noise, I, sigma

def cart2rad(N):
    """
    Generate a radial coordinate matrix.
    """
    N = int(N)
    p = (N - 1) // 2
    X, Y = np.meshgrid(np.arange(-p, p + 1), np.arange(-p, p + 1), indexing='ij')
    return np.sqrt(X**2 + Y**2)