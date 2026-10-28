#!/usr/bin/env python3
"""Panda forward kinematics in plain numpy (no ROS), for checking generated
trajectories: where the gripper goes, and where every link is.

Franka's published modified-DH parameters (Craig convention); the TCP is
TCP_OFFSET along link8 +Z, the grasp point used by pick_place.py.
"""
import math

import numpy as np

TCP_OFFSET = 0.1034  # pick_place.TCP_OFFSET
TABLE_Z = 1.025      # arm bases sit on the table top
BASES = {'first': (0.2, 0.0, 0.0), 'second': (1.5, 0.0, math.pi)}  # x, y, yaw

# a, d, alpha per joint, then the flange (link8).
DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -math.pi / 2), (0.0, 0.316, math.pi / 2),
      (0.0825, 0.0, math.pi / 2), (-0.0825, 0.384, -math.pi / 2), (0.0, 0.0, math.pi / 2),
      (0.088, 0.0, math.pi / 2)]
FLANGE_D = 0.107


def _dh(a, d, alpha, theta):
    """Modified DH: RotX(alpha) TransX(a) RotZ(theta) TransZ(d), batched over theta."""
    n = theta.shape[0]
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    T = np.zeros((n, 4, 4))
    T[:, 0, 0], T[:, 0, 1], T[:, 0, 3] = ct, -st, a
    T[:, 1, 0], T[:, 1, 1], T[:, 1, 2], T[:, 1, 3] = st * ca, ct * ca, -sa, -sa * d
    T[:, 2, 0], T[:, 2, 1], T[:, 2, 2], T[:, 2, 3] = st * sa, ct * sa, ca, ca * d
    T[:, 3, 3] = 1.0
    return T


def base_transform(arm):
    bx, by, yaw = BASES[arm]
    T = np.eye(4)
    T[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    T[:3, 3] = (bx, by, TABLE_Z)
    return T


def link_frames(q, arm=None):
    """Frames of links 1-7, the flange and the TCP for joint angles q, shape
    (..., 7). Returns (..., 9, 4, 4): in the arm's base frame, or in the world
    frame if arm is given."""
    q = np.asarray(q, dtype=float)
    shape = q.shape[:-1]
    q = q.reshape(-1, 7)
    T = np.broadcast_to(base_transform(arm) if arm else np.eye(4), (q.shape[0], 4, 4)).copy()
    frames = []
    for j, (a, d, alpha) in enumerate(DH):
        T = T @ _dh(a, d, alpha, q[:, j])
        frames.append(T)
    flange = np.eye(4)
    flange[2, 3] = FLANGE_D
    T = T @ flange
    frames.append(T)
    tcp = np.eye(4)
    tcp[2, 3] = TCP_OFFSET
    frames.append(T @ tcp)
    return np.stack(frames, axis=1).reshape(*shape, 9, 4, 4)


def tcp_position(q, arm=None):
    """Grasp point position for joint angles q (..., 7) -> (..., 3)."""
    return link_frames(q, arm)[..., -1, :3, 3]
