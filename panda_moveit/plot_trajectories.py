#!/usr/bin/env python3
"""Draw gripper (TCP) paths over the overhead scene image.

    plot_trajectories.py --seed 42 --task 0                 # both arms' best plans
    plot_trajectories.py --seed 42 --task 0 --all-plans     # every attempt, best one bold
    plot_trajectories.py --seed 42 --out /tmp/seed42.png    # all 4 tasks in a 2x2 grid

Path colour: blue for the first arm, orange for the second; a dot marks the
grasp (gripper closes) and a square the release. Usable from the ROS Python
or the LeRobot venv (numpy + OpenCV only); draw_path() is reused to overlay
generated trajectories.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from panda_kinematics import tcp_position

CAMERA = dict(x=0.85, y=0.0, z=2.6, fx=480.0, cx=480.0, cy=360.0)  # see camera.json
COLORS = {'first': (220, 120, 30), 'second': (30, 140, 255)}         # BGR
MOTION = ('pregrasp', 'descend', 'lift', 'transport', 'lower', 'retreat')


def project(points, scale=1.0):
    """World points (n, 3) -> pixel coordinates (n, 2) in an image scaled by scale.
    The camera looks straight down: image right = world +x, image down = world -y."""
    p = np.asarray(points, float)
    depth = CAMERA['z'] - p[:, 2]
    u = CAMERA['cx'] + (p[:, 0] - CAMERA['x']) * CAMERA['fx'] / depth
    v = CAMERA['cy'] - (p[:, 1] - CAMERA['y']) * CAMERA['fx'] / depth
    return np.stack([u, v], axis=1) * scale


def draw_path(image, tcp_world, color, grasp=None, release=None, thickness=2, scale=1.0):
    """Polyline of TCP positions; grasp/release are frame indices to mark."""
    px = np.round(project(tcp_world, scale)).astype(np.int32)
    cv2.polylines(image, [px.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)
    if grasp is not None:
        cv2.circle(image, tuple(map(int, px[grasp])), 5 + thickness, color, -1, cv2.LINE_AA)
    if release is not None:
        c = tuple(map(int, px[release]))
        s = 5 + thickness
        cv2.rectangle(image, (c[0] - s, c[1] - s), (c[0] + s, c[1] + s), color, -1)
    return image


def plan_path(plan, arm):
    """TCP path of a MoveIt plan, with the grasp and release indices."""
    joints, grasp, release = [], None, None
    for stage in plan['stages']:
        traj = stage.get('trajectory')
        if stage['stage'] not in MOTION or not traj:
            continue
        if stage['stage'] == 'lift':
            grasp = len(joints)          # first point after the gripper closes
        if stage['stage'] == 'retreat':
            release = len(joints)
        joints += [p[:7] for p in traj['positions']]
    return tcp_position(np.array(joints), arm), grasp, release


def draw_trial(data, trial, all_plans=False):
    image = cv2.imread(str(data / trial['rgb']))
    lines = [f"seed {trial['seed']} task {trial['task_index']}: {trial['cube']} -> {trial['tray']}"]
    for arm, record in trial['arms'].items():
        if not record['feasible']:
            lines.append(f'{arm}: infeasible')
            continue
        if all_plans:
            for i, plan in enumerate(record['plans']):
                if plan['success'] and i != record['best_plan_index']:
                    draw_path(image, plan_path(plan, arm)[0], COLORS[arm], thickness=1)
        best = record['plans'][record['best_plan_index']]
        tcp, grasp, release = plan_path(best, arm)
        draw_path(image, tcp, COLORS[arm], grasp, release, thickness=3)
        lines.append(f"{arm}: cost {best['cost']:.2f} (T {best['total_trajectory_duration']:.2f} s"
                     f" + L {best['total_joint_path_length']:.2f} rad)")
    for i, text in enumerate(lines):
        cv2.putText(image, text, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(image, text, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (20, 20, 20), 2, cv2.LINE_AA)
    return image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(Path.home() / 'panda_data'))
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--task', type=int, help='default: all tasks of the seed in a grid')
    p.add_argument('--all-plans', action='store_true')
    p.add_argument('--out', help='default: /tmp/trajectories_seed<seed>.png')
    args = p.parse_args()

    data = Path(args.data).expanduser()
    tasks = [args.task] if args.task is not None else sorted(
        int(f.stem.split('_')[-1]) for f in (data / 'trials').glob(f'seed_{args.seed}_task_*.json'))
    images = [draw_trial(data, json.loads((data / 'trials' / f'seed_{args.seed}_task_{t}.json').read_text()),
                         args.all_plans) for t in tasks]
    if len(images) == 1:
        out_image = images[0]
    else:
        images += [np.zeros_like(images[0])] * (-len(images) % 2)
        out_image = np.vstack([np.hstack(images[i:i + 2]) for i in range(0, len(images), 2)])
    out = args.out or f'/tmp/trajectories_seed{args.seed}.png'
    cv2.imwrite(out, out_image)
    print(out)


if __name__ == '__main__':
    main()
