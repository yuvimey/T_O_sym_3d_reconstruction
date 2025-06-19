import math
import numpy as np
from rotations import q_to_rot

def genRotationsGrid(resolution, JRJ=False, verbose=False):
    """
    Generate approximately equally spaced 3D rotations, similarly to the MATLAB
    function 'genRotationsGrid'.

    Parameters:
    -----------
    resolution : int
        The number of samples per 2*pi (e.g., 50, 75, 100, etc.)
    JRJ : bool, optional
        If True, apply the JRJ symmetry reduction. (Default: False)
    verbose : bool, optional
        If True, print the total number of points and status messages.

    Returns:
    --------
    rotations : numpy.ndarray
        A 3x3xN array of rotation matrices.
    angles : numpy.ndarray
        A 3xN array of angles [tau1, theta1, phi1].
    """

    pi = math.pi
    
    # First pass: Count how many rotation samples we will have
    # --------------------------------------------------------
    counter = 0
    # Step in tau1 goes from 0 -> pi/2 in (resolution/4) intervals
    tau1_step = (pi / 2.0) / (resolution / 4.0)

    if JRJ:
        # JRJ == True => theta goes from 0 -> pi/2
        # (resolution/4 * sin(tau1)) intervals
        # phi goes from 0 -> 2*pi
        # (resolution * sin(tau1) * sin(theta1)) intervals
        tau1_vals = np.arange(tau1_step/2, (pi/2) - tau1_step/2 + 1e-15, tau1_step)
        for tau1 in tau1_vals:
            stau1 = math.sin(tau1)
            # Prevent zero division if stau1 is extremely small
            if abs(stau1) < 1e-15:
                continue
            
            theta1_step = (pi/2) / (resolution/4.0 * stau1)
            theta1_vals = np.arange(theta1_step/2, (pi/2) - theta1_step/2 + 1e-15, theta1_step)
            for theta1 in theta1_vals:
                stheta1 = math.sin(theta1)
                if abs(stheta1) < 1e-15:
                    continue

                phi1_step = (2.0*pi) / (resolution * stau1 * stheta1)
                phi1_vals = np.arange(0, 2.0*pi - phi1_step/2 + 1e-15, phi1_step)
                for _ in phi1_vals:
                    counter += 1
        
        n_of_rotations = counter
        if verbose:
            print('There will be %d points', n_of_rotations)

        # Allocate arrays
        angles = np.zeros((3, n_of_rotations), dtype=float)
        rotations = np.zeros((3, 3, n_of_rotations), dtype=float)

        # Second pass: fill angles and rotations
        counter = 0
        for tau1 in tau1_vals:
            stau1 = math.sin(tau1)
            ctau1 = math.cos(tau1)
            theta1_step = (pi/2) / (resolution/4.0 * stau1)
            theta1_vals = np.arange(theta1_step/2, (pi/2) - theta1_step/2 + 1e-15, theta1_step)
            for theta1 in theta1_vals:
                stheta1 = math.sin(theta1)
                ctheta1 = math.cos(theta1)
                phi1_step = (2.0*pi) / (resolution * stau1 * stheta1)
                phi1_vals = np.arange(0, 2.0*pi - phi1_step/2 + 1e-15, phi1_step)
                for phi1 in phi1_vals:
                    angles[0, counter] = tau1
                    angles[1, counter] = theta1
                    angles[2, counter] = phi1
                    # Quaternion is [x, y, z, w]
                    q = [
                        stau1 * stheta1 * math.sin(phi1),
                        stau1 * stheta1 * math.cos(phi1),
                        stau1 * ctheta1,
                        ctau1
                    ]
                    rotations[:, :, counter] = q_to_rot([q]).reshape(3,3, order='F')
                    counter += 1

    else:
        # JRJ == False => theta goes from 0 -> pi
        # (resolution/2 * sin(tau1)) intervals
        # phi goes from 0 -> 2*pi
        # (resolution * sin(tau1) * sin(theta1)) intervals
        tau1_vals = np.arange(tau1_step/2, (pi/2) - tau1_step/2 + 1e-15, tau1_step)
        for tau1 in tau1_vals:
            stau1 = math.sin(tau1)
            if abs(stau1) < 1e-15:
                continue

            theta1_step = pi / (resolution/2.0 * stau1)
            theta1_vals = np.arange(theta1_step/2, pi - theta1_step/2 + 1e-15, theta1_step)
            for theta1 in theta1_vals:
                stheta1 = math.sin(theta1)
                if abs(stheta1) < 1e-15:
                    continue
                phi1_step = (2.0*pi) / (resolution * stau1 * stheta1)
                phi1_vals = np.arange(0, 2.0*pi - phi1_step/2 + 1e-15, phi1_step)
                for _ in phi1_vals:
                    counter += 1
        
        n_of_rotations = counter
        if verbose:
            print('There will be %d points', n_of_rotations)

        # Allocate arrays
        angles = np.zeros((3, n_of_rotations), dtype=float)
        rotations = np.zeros((3, 3, n_of_rotations), dtype=float)

        # Second pass: fill angles and rotations
        counter = 0
        for tau1 in tau1_vals:
            stau1 = math.sin(tau1)
            ctau1 = math.cos(tau1)
            theta1_step = pi / (resolution/2.0 * stau1)
            theta1_vals = np.arange(theta1_step/2, pi - theta1_step/2 + 1e-15, theta1_step)
            for theta1 in theta1_vals:
                stheta1 = math.sin(theta1)
                ctheta1 = math.cos(theta1)
                phi1_step = (2.0*pi) / (resolution * stau1 * stheta1)
                phi1_vals = np.arange(0, 2.0*pi - phi1_step/2 + 1e-15, phi1_step)
                for phi1 in phi1_vals:
                    angles[0, counter] = tau1
                    angles[1, counter] = theta1
                    angles[2, counter] = phi1
                    # Quaternion is [x, y, z, w]
                    q = [
                        stau1 * stheta1 * math.sin(phi1),
                        stau1 * stheta1 * math.cos(phi1),
                        stau1 * ctheta1,
                        ctau1
                    ]
                    rotations[:, :, counter] = q_to_rot([q]).reshape(3,3, order='F')
                    counter += 1

    return rotations, angles