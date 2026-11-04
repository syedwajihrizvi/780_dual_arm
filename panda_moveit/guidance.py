#!/usr/bin/env python3
"""Cost guidance for diffusion sampling: steer samples away from obstacles
and towards the cube and the place target while they are denoised.

The cost is computed in joint space with differentiable forward kinematics
and the sphere model of collision_check.py (same radii), so the guidance
pushes on what the checker measures:
    obstacle  sum over frames and spheres of relu(margin - gap)^2, against the
              obstacle boxes, tray walls, table and the other arm (static)
    goal      squared TCP distance to the cube around the frame where the
              gripper closes, and to the place target where it opens
    smooth    squared second differences of the correction (guided minus
              the model's prediction), so corrections stay smooth without
              flattening the motion the model predicted

Sampler.chunk (evaluate_policy.py) calls Guide.cost on the predicted clean
trajectory between DDIM steps; nothing is backpropagated through the U-Net.
"""
import math

import numpy as np
import torch

from collision_check import FLANGE_RADIUS, HAND_POINTS, HAND_RADIUS, LINK_RADII, SPHERE_STEP, \
    TABLE_EXEMPT_JOINTS, arm_spheres, scene_boxes
from panda_kinematics import DH, FLANGE_D, TABLE_Z, TCP_OFFSET, base_transform

ARMS = ('first', 'second')


def _dh(a, d, alpha, theta):
    """Modified DH transforms for theta (...,) -> (..., 4, 4)."""
    ct, st = torch.cos(theta), torch.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    zero, one = torch.zeros_like(theta), torch.ones_like(theta)
    rows = [torch.stack([ct, -st, zero, a * one], -1),
            torch.stack([st * ca, ct * ca, -sa * one, -sa * d * one], -1),
            torch.stack([st * sa, ct * sa, ca * one, ca * d * one], -1),
            torch.stack([zero, zero, zero, one], -1)]
    return torch.stack(rows, -2)


def link_frames(q, base):
    """Frames of links 1-7, the flange and the TCP (..., 9, 4, 4) for joints
    q (..., 7), in the world frame given the arm's base transform (4, 4)."""
    T = base.expand(*q.shape[:-1], 4, 4)
    frames = []
    for j, (a, d, alpha) in enumerate(DH):
        T = T @ _dh(a, d, alpha, q[..., j])
        frames.append(T)
    offset = torch.eye(4, dtype=q.dtype, device=q.device)
    flange = offset.clone()
    flange[2, 3] = FLANGE_D
    T = T @ flange
    frames.append(T)
    tcp = offset.clone()
    tcp[2, 3] = TCP_OFFSET
    frames.append(T @ tcp)
    return torch.stack(frames, -3)


def _weights():
    """Each sphere centre as a weighted sum of key points (s, K): the base,
    then each joint's corner and frame origin, the flange, the hand points;
    spheres are spaced along each leg as in collision_check.arm_spheres."""
    rows = []

    def leg(start, end, length):
        if length < 1e-6:
            return
        for w in np.linspace(0.0, 1.0, max(2, int(math.ceil(length / SPHERE_STEP)) + 1)):
            row = np.zeros(16 + len(HAND_POINTS))
            row[start] += 1 - w
            row[end] += w
            rows.append(row)

    for j, (a, d, _) in enumerate(DH):
        start = 0 if j == 0 else 2 * j
        leg(start, 2 * j + 1, abs(a))
        leg(2 * j + 1, 2 * j + 2, abs(d))
    leg(14, 15, FLANGE_D)
    for k in range(len(HAND_POINTS)):
        row = np.zeros(16 + len(HAND_POINTS))
        row[16 + k] = 1.0
        rows.append(row)
    return np.array(rows)


WEIGHTS = _weights()


def spheres(q, base, weights):
    """Sphere centres (..., s, 3) along the links, as collision_check.arm_spheres,
    and the TCP (..., 3). weights: WEIGHTS as a tensor."""
    frames = link_frames(q, base)
    previous = base.expand(*q.shape[:-1], 4, 4)
    points = [previous[..., :3, 3]]
    for j, (a, _, _) in enumerate(DH):
        points += [previous[..., :3, 3] + a * previous[..., :3, 0], frames[..., j, :3, 3]]
        previous = frames[..., j, :, :]
    points.append(frames[..., 7, :3, 3])
    flange = frames[..., 7, :, :]
    hand = torch.tensor(HAND_POINTS, dtype=q.dtype, device=q.device)
    points = torch.cat([torch.stack(points, -2),
                        hand @ flange[..., :3, :3].transpose(-1, -2) + flange[..., None, :3, 3]], -2)
    return weights @ points, frames[..., -1, :3, 3]


def _radii():
    """Radii (s,) and table exemption (s,) in the order spheres() makes them."""
    radii, exempt = [], []
    for j, ((a, d, _), (r_a, r_d)) in enumerate(zip(DH, LINK_RADII)):
        for length, r in ((abs(a), r_a), (abs(d), r_d)):
            if length >= 1e-6:
                n = max(2, int(math.ceil(length / SPHERE_STEP)) + 1)
                radii += [r] * n
                exempt += [j < TABLE_EXEMPT_JOINTS] * n
    n = max(2, int(math.ceil(FLANGE_D / SPHERE_STEP)) + 1)
    radii += [FLANGE_RADIUS] * n
    exempt += [False] * n
    radii += [HAND_RADIUS] * len(HAND_POINTS)
    exempt += [True] * len(HAND_POINTS)
    return np.array(radii), np.array(exempt)


class Guide:
    """Guidance cost for one trial's samples. Rows of a batch are (arm, phase)
    pairs; phase 0 is before the grasp, 1 carrying, 2 after the release."""

    def __init__(self, scenario, trial, device, obstacle=1.0, goal=1.0, smooth=0.1, margin=0.02,
                 gripper_mid=0.0275):
        self.obstacle, self.goal, self.smooth, self.margin = obstacle, goal, smooth, margin
        self.gripper_mid = gripper_mid
        t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
        self.base = {a: t(base_transform(a)) for a in ARMS}
        self.weights = t(WEIGHTS)
        radii, exempt = _radii()
        self.radii, self.exempt = t(radii), torch.as_tensor(exempt, device=device)
        boxes = scene_boxes(scenario)
        self.box_centre = t([b[1] for b in boxes])                       # (b, 3)
        self.box_cos = t([math.cos(b[2]) for b in boxes])
        self.box_sin = t([math.sin(b[2]) for b in boxes])
        self.box_half = t([b[3] for b in boxes])
        others = []
        for a in ARMS:
            other = ARMS[1 - ARMS.index(a)]
            centres, self.other_radii, _ = arm_spheres(np.asarray(scenario['arm_starts'][other], float)[None], other)
            others.append(centres[0])
        self.other_centres, self.other_radii = t(others), t(self.other_radii)  # (2, s, 3), (s,)
        self.bases = torch.stack([self.base[a] for a in ARMS])            # (2, 4, 4)
        self.targets = torch.stack([t(trial['cube_pose']['xyz']), t(trial['place_xyz'])])

    def cost(self, q, start, arms, phases, reference):
        """Cost per row (B,) of chunks q (B, H, 8), each following its start
        state (B, 8); reference (B, H, 8) is the model's unguided prediction.
        arms and phases: per row."""
        traj = torch.cat([start[:, None], q], 1)                          # (B, H+1, 8)
        joints = traj[..., :7]
        total = torch.zeros(len(q), device=q.device)
        if self.obstacle or self.goal:
            arm = torch.tensor([ARMS.index(a) for a in arms], device=q.device)
            centres, tcp = spheres(joints, self.bases[arm][:, None], self.weights)  # (B, H+1, s, 3), (B, H+1, 3)
            if self.obstacle:
                total = total + self.obstacle * self._obstacle(centres, arm)
            if self.goal:
                total = total + self.goal * self._goal(tcp, traj[..., 7], phases)
        if self.smooth:
            change = torch.cat([torch.zeros_like(start[:, None, :7]), (q - reference)[..., :7]], 1)
            total = total + self.smooth * (change[:, 2:] - 2 * change[:, 1:-1] + change[:, :-2]).pow(2).sum((1, 2))
        return total

    def _obstacle(self, centres, arm):
        d = centres[..., None, :] - self.box_centre                       # (B, H, s, b, 3)
        x = self.box_cos * d[..., 0] + self.box_sin * d[..., 1]
        y = -self.box_sin * d[..., 0] + self.box_cos * d[..., 1]
        local = torch.stack([x, y, d[..., 2]], -1)
        outside = torch.clamp(local.abs() - self.box_half, min=0.0)
        gap = torch.sqrt(outside.pow(2).sum(-1) + 1e-9) - self.radii[:, None]  # (B, H, s, b)
        cost = torch.relu(self.margin - gap).pow(2).sum((-1, -2, -3))
        table = centres[..., 2] - TABLE_Z - self.radii
        cost = cost + torch.where(self.exempt, 0.0, torch.relu(self.margin - table).pow(2)).sum((-1, -2))
        gap = torch.cdist(centres.reshape(len(centres), -1, 3), self.other_centres[arm])
        gap = gap.reshape(*centres.shape[:-1], -1) - self.radii[:, None] - self.other_radii
        return cost + torch.relu(self.margin - gap).pow(2).sum((-1, -2, -3))

    def _goal(self, tcp, gripper, phases):
        """TCP to the cube around the first closing frame (phase 0) and to the
        place target around the first opening frame after it."""
        closed = (gripper < self.gripper_mid).cpu().numpy()
        rows, frames, targets = [], [], []
        for i, phase in enumerate(phases):
            if phase == 0:
                shut = np.flatnonzero(closed[i, 1:]) + 1
                if len(shut):
                    rows.append(i), frames.append(shut[0]), targets.append(0)
                    opened = np.flatnonzero(~closed[i, shut[0]:]) + shut[0]
                    if len(opened):
                        rows.append(i), frames.append(opened[0]), targets.append(1)
            elif phase == 1:
                opened = np.flatnonzero(~closed[i, 1:]) + 1
                if len(opened):
                    rows.append(i), frames.append(opened[0]), targets.append(1)
        cost = torch.zeros(len(tcp), device=tcp.device)
        if not rows:
            return cost
        rows = torch.tensor(rows, device=tcp.device)
        frames = torch.tensor(frames, device=tcp.device)
        target = self.targets[torch.tensor(targets, device=tcp.device)]
        # The gripper acts between the frame before the switch and the switch frame.
        error = torch.stack([(tcp[rows, torch.clamp(frames - 1, min=1)] - target).pow(2).sum(-1),
                             (tcp[rows, frames] - target).pow(2).sum(-1)]).min(0).values
        return cost.index_add(0, rows, error)
