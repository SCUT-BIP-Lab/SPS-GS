#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
import math
from scene.cameras import Camera

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
def matrix_to_quaternion(matrix):
    """
    Convert a rotation matrix to a quaternion.
    
    Args:
        matrix (torch.Tensor): A batch of rotation matrices of shape (N, 3, 3).
        
    Returns:
        torch.Tensor: A batch of quaternions of shape (N, 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    # Symmetric choice for diagonal
    t = m00 + m11 + m22
    
    # Initialize output tensor
    q = torch.empty(batch_dim + (4,), dtype=matrix.dtype, device=matrix.device)

    # Condition for the diagonal term
    # t > 0
    s_cond1 = t > 0
    if torch.any(s_cond1):
        t_cond1 = t[s_cond1]
        s = 0.5 / torch.sqrt(t_cond1 + 1.0)
        q[s_cond1, 0] = 0.25 / s
        q[s_cond1, 1] = (m21[s_cond1] - m12[s_cond1]) * s
        q[s_cond1, 2] = (m02[s_cond1] - m20[s_cond1]) * s
        q[s_cond1, 3] = (m10[s_cond1] - m01[s_cond1]) * s

    # All other cases
    s_cond_other = ~s_cond1
    if torch.any(s_cond_other):
        # m00 > m11 and m00 > m22
        cond2 = (m00 > m11) & (m00 > m22)
        s_cond2 = s_cond_other & cond2
        if torch.any(s_cond2):
            t_cond2 = t[s_cond2]
            s = 2.0 * torch.sqrt(1.0 + m00[s_cond2] - m11[s_cond2] - m22[s_cond2])
            q[s_cond2, 0] = (m21[s_cond2] - m12[s_cond2]) / s
            q[s_cond2, 1] = 0.25 * s
            q[s_cond2, 2] = (m01[s_cond2] + m10[s_cond2]) / s
            q[s_cond2, 3] = (m02[s_cond2] + m20[s_cond2]) / s

        # m11 > m22
        cond3 = ~cond2 & (m11 > m22)
        s_cond3 = s_cond_other & cond3
        if torch.any(s_cond3):
            t_cond3 = t[s_cond3]
            s = 2.0 * torch.sqrt(1.0 + m11[s_cond3] - m00[s_cond3] - m22[s_cond3])
            q[s_cond3, 0] = (m02[s_cond3] - m20[s_cond3]) / s
            q[s_cond3, 1] = (m01[s_cond3] + m10[s_cond3]) / s
            q[s_cond3, 2] = 0.25 * s
            q[s_cond3, 3] = (m12[s_cond3] + m21[s_cond3]) / s

        # Else
        cond4 = ~cond2 & ~cond3
        s_cond4 = s_cond_other & cond4
        if torch.any(s_cond4):
            t_cond4 = t[s_cond4]
            s = 2.0 * torch.sqrt(1.0 + m22[s_cond4] - m00[s_cond4] - m11[s_cond4])
            q[s_cond4, 0] = (m10[s_cond4] - m01[s_cond4]) / s
            q[s_cond4, 1] = (m02[s_cond4] + m20[s_cond4]) / s
            q[s_cond4, 2] = (m12[s_cond4] + m21[s_cond4]) / s
            q[s_cond4, 3] = 0.25 * s

    return q

def slerp(q1, q2, t, DOT_THRESHOLD=0.9995):
    """
    Performs Spherical Linear Interpolation (SLERP) between two quaternions.
    
    Args:
        q1 (np.ndarray): Start quaternion (w, x, y, z).
        q2 (np.ndarray): End quaternion (w, x, y, z).
        t (float): Interpolation factor (0 to 1).
        DOT_THRESHOLD (float): Threshold for near-parallel quaternions.
    
    Returns:
        np.ndarray: Interpolated quaternion.
    """
    # Asegúrate de que los cuaterniones estén normalizados
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    dot = np.dot(q1, q2)

    # Si los cuaterniones están en direcciones opuestas, invierte uno para tomar la ruta más corta.
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    if dot > DOT_THRESHOLD:
        # Si están muy cerca, usa interpolación lineal y normaliza.
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    # SLERP
    theta_0 = np.arccos(dot)        # Ángulo entre cuaterniones
    theta = theta_0 * t             # Ángulo a interpolar
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    
    return (s1 * q1) + (s2 * q2)


def interpolate_cameras(cam1: Camera, cam2: Camera, t: float):
    """
    Interpolates between two camera objects, now correctly handling the K matrix.
    
    Args:
        cam1 (Camera): The starting camera.
        cam2 (Camera): The ending camera.
        t (float): Interpolation factor (0 to 1).
        
    Returns:
        Camera: A new, interpolated camera object.
    """
    # 1. Interpolate position (camera center) linearly
    center1 = -np.dot(cam1.R.T, cam1.T)
    center2 = -np.dot(cam2.R.T, cam2.T)
    new_center = (1 - t) * center1 + t * center2

    # 2. Interpolate rotation using SLERP on quaternions
    from scipy.spatial.transform import Rotation as R
    q1_scipy = R.from_matrix(cam1.R).as_quat()
    q2_scipy = R.from_matrix(cam2.R).as_quat()
    
    q1 = np.array([q1_scipy[3], q1_scipy[0], q1_scipy[1], q1_scipy[2]])
    q2 = np.array([q2_scipy[3], q2_scipy[0], q2_scipy[1], q2_scipy[2]])
    
    interp_q = slerp(q1, q2, t)

    interp_q_scipy = np.array([interp_q[1], interp_q[2], interp_q[3], interp_q[0]])
    new_R = R.from_quat(interp_q_scipy).as_matrix()

    # 3. Calculate new translation vector T
    new_T = -np.dot(new_R, new_center)
    
    # 4. Interpolate FoV
    new_FoVx = (1 - t) * cam1.FoVx + t * cam2.FoVx
    new_FoVy = (1 - t) * cam1.FoVy + t * cam2.FoVy

    # --- THIS IS THE FIX ---
    # 5. Construct the new K matrix from the interpolated FoV
    # We assume the image dimensions are the same as cam1
    image_width = cam1.image_width
    image_height = cam1.image_height
    
    focal_x = image_width / (2.0 * math.tan(new_FoVx * 0.5))
    focal_y = image_height / (2.0 * math.tan(new_FoVy * 0.5))
    
    new_K = np.array([
        [focal_x, 0, image_width / 2.0],
        [0, focal_y, image_height / 2.0],
        [0, 0, 1.0]
    ])
    # --- END FIX ---

    # 6. Create a new Camera object with the new K matrix
    new_cam = Camera(
        colmap_id=-1,
        R=new_R, 
        T=new_T, 
        FoVx=new_FoVx, 
        FoVy=new_FoVy, 
        image=cam1.original_image, # Use a placeholder image
        gt_alpha_mask=None,
        image_name=f"interp_{cam1.uid}_{cam2.uid}_{t:.2f}", 
        uid=-1,
        K=new_K  # Pass the newly constructed K matrix
    )
    
    # The Camera __init__ should now correctly set all other attributes like tanfovx
    return new_cam
