#!/usr/bin/env python3
"""Fast approximate collision check for a joint trajectory in a scenario.

Each arm is a chain of spheres along its links (from panda_kinematics); the
scene is the obstacle boxes, the tray walls, the table top and the other arm
at its start pose. Meant for screening generated trajectories in bulk, not
as a replacement for MoveIt: radii are tuned so MoveIt's own collision-free
plans pass (see --validate).

    collision_check.py --validate            # check MoveIt's best plans
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from panda_kinematics import DH, TABLE_Z, base_transform, link_frames

# Each joint moves the chain a along the previous frame's x axis, then d along
# the new z axis (modified DH), so links are L-shaped: the spheres follow both
# legs. Radii per joint, (a leg, d leg); then the flange.
LINK_RADII = [(0.08, 0.08), (0.07, 0.07), (0.07, 0.07), (0.065, 0.065),
              (0.065, 0.055), (0.055, 0.055), (0.055, 0.055)]
FLANGE_RADIUS = 0.05
SPHERE_STEP = 0.04
# The palm only (flange frame, ~0.16 m wide along y); the thin fingers below it
# reach into trays and around the cube by design, so they are not modelled.
HAND_RADIUS = 0.03
HAND_POINTS = [(0.0, y, z) for y in (-0.05, 0.0, 0.05) for z in (0.0, 0.035)]
TABLE_EXEMPT_JOINTS = 2   # base and shoulder sit on the table by design
TRAY_WALL = (0.01, 0.03)   # thickness, height; outer size from the scenario
EXACT_BELOW = 0.15         # m; other-arm gaps are exact below this, a lower bound above


def arm_spheres(q, arm):
    """Sphere centres (n, s, 3), radii (s,) and whether each sphere is exempt
    from the table check (s,), for joint trajectory q (n, 7)."""
    q = np.asarray(q, float)
    frames = link_frames(q, arm)                                  # (n, 9, 4, 4)
    previous = np.broadcast_to(base_transform(arm), (len(q), 4, 4))
    centres, radii, table_ok = [], [], []

    def leg(start, end, radius, exempt):
        length = np.linalg.norm(end[0] - start[0])
        if length < 1e-6:
            return
        for w in np.linspace(0.0, 1.0, max(2, int(math.ceil(length / SPHERE_STEP)) + 1)):
            centres.append(start * (1 - w) + end * w)
            radii.append(radius)
            table_ok.append(exempt)

    for j, ((a, _, _), (r_a, r_d)) in enumerate(zip(DH, LINK_RADII)):
        start = previous[:, :3, 3]
        corner = start + a * previous[:, :3, 0]
        exempt = j < TABLE_EXEMPT_JOINTS
        leg(start, corner, r_a, exempt)
        leg(corner, frames[:, j, :3, 3], r_d, exempt)
        previous = frames[:, j]
    leg(frames[:, 6, :3, 3], frames[:, 7, :3, 3], FLANGE_RADIUS, False)
    flange = frames[:, 7]
    for p in HAND_POINTS:
        centres.append((flange @ np.array([*p, 1.0]))[:, :3])
        radii.append(HAND_RADIUS)
        table_ok.append(True)  # the hand reaches down to the table to grasp
    return np.stack(centres, axis=1), np.array(radii), np.array(table_ok)


def scene_boxes(scenario):
    """Oriented boxes (name, centre xyz, yaw, half sizes) for obstacles and tray walls."""
    boxes = [(o['name'], np.array(o['xyz']), o['yaw'], np.array(o['size']) / 2)
             for o in scenario['obstacles']]
    thickness, height = TRAY_WALL
    for tray in scenario['trays']:
        half = tray['size'][0] / 2
        c, s = math.cos(tray['yaw']), math.sin(tray['yaw'])
        for dx, dy, hx, hy in ((half - thickness / 2, 0, thickness / 2, half),
                               (-(half - thickness / 2), 0, thickness / 2, half),
                               (0, half - thickness / 2, half, thickness / 2),
                               (0, -(half - thickness / 2), half, thickness / 2)):
            centre = np.array([tray['xyz'][0] + c * dx - s * dy, tray['xyz'][1] + s * dx + c * dy,
                               TABLE_Z + height / 2])
            boxes.append((tray['name'], centre, tray['yaw'], np.array([hx, hy, height / 2])))
    return boxes


def box_distance(points, centre, yaw, half):
    """Distance from points (..., 3) to an oriented box (0 inside)."""
    d = points - centre
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], d[..., 2]], -1)
    outside = np.maximum(np.abs(local) - half, 0.0)
    return np.linalg.norm(outside, axis=-1)


def check(q, arm, scenario, other_q=None):
    """Clearance of a trajectory q (n, 7): smallest gap between the arm's
    spheres and anything in the scene (negative = collision), with the
    frame and object where it happens."""
    return check_many([q], arm, scenario, other_q)[0]


def check_many(trajectories, arm, scenario, other_q=None):
    """check() for several trajectories of one arm in one vectorized pass."""
    lengths = [len(q) for q in trajectories]
    q = np.concatenate([np.asarray(t, float).reshape(-1, 7) for t in trajectories])
    centres, radii, table_ok = arm_spheres(q, arm)
    results = []
    for name, centre, yaw, half in scene_boxes(scenario):
        gap = box_distance(centres, centre, yaw, half) - radii
        results.append((gap, name))
    gap = centres[..., 2] - TABLE_Z - radii
    results.append((np.where(table_ok, np.inf, gap), 'table'))
    other = 'second' if arm == 'first' else 'first'
    oq = np.asarray(other_q if other_q is not None else scenario['arm_starts'][other], float)
    oc, orad, _ = arm_spheres(oq[None], other)
    oc, orad = oc[0], orad
    # The other arm is static. Distance to the bounding box of its sphere
    # centres, minus both radii, is a lower bound on the gap; only spheres
    # where that bound is small get exact pairwise distances (gaps above
    # EXACT_BELOW are reported as the bound).
    lo, hi = oc.min(0), oc.max(0)
    outside = np.maximum(np.maximum(lo - centres, centres - hi), 0.0)
    gap = np.linalg.norm(outside, axis=-1) - radii - orad.max()
    near = np.flatnonzero((gap < EXACT_BELOW).reshape(-1))
    if len(near):
        flat = centres.reshape(-1, 3)[near]
        r = np.broadcast_to(radii, centres.shape[:2]).reshape(-1)[near]
        d = np.linalg.norm(flat[:, None] - oc[None], axis=-1) - r[:, None] - orad[None]
        gap.reshape(-1)[near] = d.min(axis=1)
    results.append((gap, f'{other}_arm'))

    per_object = np.stack([g.min(axis=1) for g, _ in results])            # (objects, frames)
    names = [name for _, name in results]
    out, start = [], 0
    for n in lengths:
        block = per_object[:, start:start + n]
        obj, frame = np.unravel_index(block.argmin(), block.shape)
        clearance = block[obj, frame]
        out.append(dict(clearance=float(clearance), collision=bool(clearance < 0),
                        object=names[obj], frame=int(frame)))
        start += n
    return out


def plan_joints(plan):
    joints = []
    for stage in plan['stages']:
        traj = stage.get('trajectory')
        if traj and stage['stage'] != 'grasp_select':
            joints += [p[:7] for p in traj['positions']]
    return np.array(joints)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(Path.home() / 'panda_data'))
    p.add_argument('--validate', type=int, default=300, metavar='N', help='check the first N trials')
    args = p.parse_args()
    data = Path(args.data).expanduser()
    lines = [json.loads(l) for l in (data / 'trials.jsonl').read_text().splitlines() if l][:args.validate]
    clear, hits = [], []
    for line in lines:
        trial = json.loads((data / line['trial']).read_text())
        scenario = json.loads((data / trial['scenario']).read_text())
        for arm, record in trial['arms'].items():
            if not record['feasible']:
                continue
            r = check(plan_joints(record['plans'][record['best_plan_index']]), arm, scenario)
            clear.append(r['clearance'])
            if r['collision']:
                hits.append((trial['seed'], trial['task_index'], arm, r['object'], round(r['clearance'], 3)))
    clear = np.array(clear)
    print(f'MoveIt best plans checked: {len(clear)}; flagged as colliding: {len(hits)} '
          f'({100 * len(hits) / len(clear):.1f}%)')
    print('clearance m: ' + '  '.join(f'p{q} {np.percentile(clear, q):.3f}' for q in (1, 5, 25, 50)))
    from collections import Counter
    print('flagged objects:', Counter(h[3].rsplit('_', 1)[0] if h[3].startswith('obstacle') else h[3]
                                      for h in hits).most_common())
    for h in hits[:8]:
        print('  ', h)


if __name__ == '__main__':
    main()
