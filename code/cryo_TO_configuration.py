def cryo_TO_configuration():
    """
    Configuration settings for TO symmetry processing.
    
    Returns:
        n_theta: int
            Number of radial lines.
        n_r: int
            Number of equispaced samples along each radial line.
        rmax: int
            Maximum radius.
        resolution: int
            Number of samples per 2*pi (for genRotationsGrid).
        viewing_angle: float
            Viewing angle threshold.
        inplane_rot_degree: int
            In-plane rotation degree threshold.
    """
    n_theta = 360 # L radial lines.
    n_r =  89  # Number of equispaced samples along each radial line.
    rmax = 2  
    
    resolution = 150  # Number of samples per 2*pi.
    viewing_angle = 0.996  # Viewing angle threshold.
    inplane_rot_degree = 5  # In-plane rotation degree threshold.
    
    return n_theta, n_r, rmax, resolution, viewing_angle, inplane_rot_degree