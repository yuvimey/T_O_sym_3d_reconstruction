import numpy as np

def cart2pol(x, y):
    phi = np.arctan2(y, x)
    r = np.hypot(x, y)
    return phi, r

def mesh_2d(L, inclusive=False):
    """
    Define an image mesh and mask.

    Parameters:
        L (int): The dimension of the desired mesh.
        inclusive (bool): If True, both endpoints -1 and +1 are included.
                          Defaults to False.

    Returns:
        dict: Contains fields:
            - 'x', 'y': x and y coordinates in an L-by-L grid.
            - 'r', 'phi': Polar coordinates corresponding to (x,y).
    """
    if not inclusive:
        grid = np.ceil(np.arange(-L/2, L/2)) / (L / 2)
    else:
        grid = np.arange(-(L - 1) / 2, (L + 1) / 2) / ((L - 1) / 2)

    mesh_x, mesh_y = np.meshgrid(grid, grid, indexing='ij')
    mesh_phi, mesh_r = cart2pol(mesh_x, mesh_y)

    return {'x': mesh_x, 'y': mesh_y, 'r': mesh_r, 'phi': mesh_phi}

def rotated_grids(L, rot_matrices, half_pixel=False):
    """
    Generate rotated Fourier grids in 3D from rotation matrices.
    
    Parameters:
        L (int): The resolution of the desired grids.
        rot_matrices (np.ndarray): An array of shape (3, 3, K) containing K rotation matrices.
        half_pixel (bool, optional): If True, centers the rotation around a half-pixel (default False).
    
    Returns:
        np.ndarray: A set of rotated Fourier grids in three dimensions as specified by the rotation matrices.
    """
    mesh2d = mesh_2d(L)
    
    if L % 2 == 0 and half_pixel:
        mesh2d['x'] += 1 / L
        mesh2d['y'] += 1 / L
    
    num_pts = L**2
    num_rots = rot_matrices.shape[2]
    
    pts = np.pi * np.vstack([mesh2d['x'].ravel(order='F'), mesh2d['y'].ravel(order='F'), np.zeros(num_pts)])
    
    pts_rot = np.zeros((3, num_pts, num_rots))
    
    for s in range(num_rots):
        pts_rot[:, :, s] = rot_matrices[:, :, s] @ pts
    
    pts_rot = pts_rot.reshape(3, L, L, num_rots, order='F')
    
    return pts_rot