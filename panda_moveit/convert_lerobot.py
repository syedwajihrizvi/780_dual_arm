#!/usr/bin/env python3
"""Convert collect.py output to LeRobot datasets for Diffusion Policy.

One episode = one arm's best feasible plan for one task, resampled to a
fixed rate. Both arms go into the same datasets (one shared model): every
input is expressed in the acting arm's base frame, the overhead image is
rotated 180 degrees for the second arm so each arm sees the table from its
own side, and an arm ID is appended to the environment state.

Writes one dataset per split (split by seed, never by task):
    <out>/train, <out>/val, <out>/test   LeRobot datasets
    <out>/<split>_episodes.jsonl         episode index -> seed, task, arm, MoveIt cost
    <out>/infeasible.jsonl               (trial, arm) pairs with no feasible plan
    <out>/conversion.json                settings and environment_state field names

Run with the LeRobot venv, not the ROS Python:
    ~/lerobot_venv/bin/python convert_lerobot.py --data ~/panda_data --out ~/lerobot_data
    ~/lerobot_venv/bin/python convert_lerobot.py --data ~/panda_data --out /tmp/lr --max-trials 20
"""
import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Must match the collection code (pick_place.py, scenario.py, the world file).
GRIPPER_OPEN = 0.04
GRIPPER_CLOSE = 0.015
TABLE_Z = 1.025                                   # arm bases sit on the table top
BASES = {'first': (0.2, 0.0, 0.0), 'second': (1.5, 0.0, math.pi)}  # x, y, yaw
CAMERA_Z = 2.6
MAX_OBSTACLES = 6                                 # scenario.OBSTACLE_COUNT upper bound

MOTION = {'pregrasp': GRIPPER_OPEN, 'descend': GRIPPER_OPEN, 'lift': GRIPPER_CLOSE,
          'transport': GRIPPER_CLOSE, 'lower': GRIPPER_CLOSE, 'retreat': GRIPPER_OPEN}
SPLITS = {'val': (1700, 1850), 'test': (1850, 2000)}  # everything else trains
IMAGE_SIZE = (320, 240)                           # width, height; walls vanish below this
MAX_HEIGHT = 0.35                                 # height map clip, above the tallest obstacle

ARM_IDS = {'first': [1.0, 0.0], 'second': [0.0, 1.0]}
OTHER = {'first': 'second', 'second': 'first'}


def split_of(seed):
    return next((name for name, (lo, hi) in SPLITS.items() if lo <= seed < hi), 'train')


# -- Arm base frame ---------------------------------------------------

def to_base(arm, xyz):
    """World point -> the arm's base frame (x forward, z up from the table)."""
    bx, by, byaw = BASES[arm]
    dx, dy = xyz[0] - bx, xyz[1] - by
    c, s = math.cos(byaw), math.sin(byaw)
    return [c * dx + s * dy, -s * dx + c * dy, xyz[2] - TABLE_Z]


def yaw_features(arm, yaw, symmetry):
    """sin/cos of a yaw in the arm's frame. Squares look the same every 90
    degrees and boxes every 180, so the angle is multiplied by that order;
    this also makes the 180-degree base rotation of the second arm drop out."""
    a = symmetry * (yaw - BASES[arm][2])
    return [math.sin(a), math.cos(a)]


ENV_NAMES = (['cube_x', 'cube_y', 'cube_z', 'cube_sin4yaw', 'cube_cos4yaw',
              'place_x', 'place_y', 'place_z', 'tray_sin4yaw', 'tray_cos4yaw',
              'other_tray_x', 'other_tray_y', 'other_tray_sin4yaw', 'other_tray_cos4yaw']
             + [f'obstacle{i}_{f}' for i in range(MAX_OBSTACLES)
                for f in ('x', 'y', 'sin2yaw', 'cos2yaw', 'length', 'width', 'height', 'present')]
             + [f'other_arm_joint{i}' for i in range(1, 8)]
             + ['arm_is_first', 'arm_is_second'])


def environment_state(scenario, trial, arm):
    """Task and scene for one arm, in its base frame (see ENV_NAMES)."""
    cube = next(c for c in scenario['cubes'] if c['name'] == trial['cube'])
    tray = next(t for t in scenario['trays'] if t['name'] == trial['tray'])
    other_tray = next(t for t in scenario['trays'] if t['name'] != trial['tray'])
    v = to_base(arm, cube['xyz']) + yaw_features(arm, cube['yaw'], 4)
    v += to_base(arm, trial['place_xyz']) + yaw_features(arm, tray['yaw'], 4)
    v += to_base(arm, other_tray['xyz'])[:2] + yaw_features(arm, other_tray['yaw'], 4)
    # Nearest obstacles first, so slot order means something.
    obstacles = sorted(scenario['obstacles'],
                       key=lambda o: math.hypot(*to_base(arm, o['xyz'])[:2]))
    for i in range(MAX_OBSTACLES):
        if i < len(obstacles):
            o = obstacles[i]
            v += to_base(arm, o['xyz'])[:2] + yaw_features(arm, o['yaw'], 2) + list(o['size']) + [1.0]
        else:
            v += [0.0] * 8
    v += list(scenario['arm_starts'][OTHER[arm]])   # the other arm stays put, an obstacle
    v += ARM_IDS[arm]
    assert len(v) == len(ENV_NAMES)
    return np.asarray(v, np.float32)


# -- Trajectories -----------------------------------------------------

def episode_states(plan, fps):
    """Join the motion stages and resample to a fixed rate: (n, 8) of
    7 joints + gripper width. The gripper is a step function per stage."""
    times, joints, grip, t0 = [], [], [], 0.0
    for stage in plan['stages']:
        traj = stage.get('trajectory')
        if stage['stage'] not in MOTION or not traj:
            continue
        t = np.asarray(traj['time_from_start']) + t0
        times.append(t)
        joints.append(np.asarray(traj['positions'])[:, :7])
        grip.append(np.full(len(t), MOTION[stage['stage']]))
        t0 = t[-1]
    t, q, g = map(np.concatenate, (times, joints, grip))
    t, keep = np.unique(t, return_index=True)   # drop the repeated point at each join
    q, g = q[keep], g[keep]
    # Fixed rate, so frames / fps is the trajectory time; the last frame
    # reaches the final point (np.interp holds it past the end).
    grid = np.arange(0.0, t[-1] + 1.0 / fps - 1e-9, 1.0 / fps)
    q_rs = np.stack([np.interp(grid, t, q[:, j]) for j in range(7)], axis=1)
    g_rs = g[np.searchsorted(t, grid, side='right') - 1]
    return np.hstack([q_rs, g_rs[:, None]]).astype(np.float32)


# -- Images -----------------------------------------------------------

class SceneImages:
    """Before images of one seed, resized, for each arm's point of view."""

    def __init__(self, data, trial, height_map):
        rgb = cv2.cvtColor(cv2.imread(str(data / trial['rgb'])), cv2.COLOR_BGR2RGB)
        self.rgb = cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        self.height = None
        if height_map:
            depth = np.load(data / trial['depth'])['depth']
            # Height above the table; background (inf depth) and the table read 0.
            h = np.clip((CAMERA_Z - TABLE_Z) - depth, 0.0, MAX_HEIGHT)
            h = cv2.resize((h / MAX_HEIGHT * 255).astype(np.uint8), IMAGE_SIZE,
                           interpolation=cv2.INTER_AREA)
            self.height = np.repeat(h[:, :, None], 3, axis=2)

    def view(self, image, arm):
        # The camera sits midway between the bases, above the image centre, so
        # a 180-degree turn puts the second arm where the first one is.
        return image if arm == 'first' else cv2.rotate(image, cv2.ROTATE_180)


# -- Dataset ----------------------------------------------------------

def features(height_map):
    w, h = IMAGE_SIZE
    image = {'shape': (h, w, 3), 'names': ['height', 'width', 'channels']}
    state_names = [f'joint{i}' for i in range(1, 8)] + ['gripper_width']
    f = {'observation.state': {'dtype': 'float32', 'shape': (8,), 'names': state_names},
         'observation.environment_state': {'dtype': 'float32', 'shape': (len(ENV_NAMES),),
                                           'names': ENV_NAMES},
         'observation.images.overhead': {'dtype': 'video', **image},
         'action': {'dtype': 'float32', 'shape': (8,), 'names': state_names}}
    if height_map:
        # Not video: MP4 compression would corrupt the height values.
        f['observation.images.height'] = {'dtype': 'image', **image}
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(Path.home() / 'panda_data'))
    p.add_argument('--out', default=str(Path.home() / 'lerobot_data'))
    p.add_argument('--repo-prefix', default='dual_panda', help='repo_id is <prefix>_<split>')
    p.add_argument('--fps', type=int, default=5)
    p.add_argument('--arms', nargs='+', default=['first', 'second'],
                   help='one arm only gives a per-arm dataset')
    p.add_argument('--height-map', action='store_true', help='phase 2: add the depth height map')
    p.add_argument('--max-trials', type=int, help='convert only the first N trials (testing)')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    data, out = Path(args.data).expanduser(), Path(args.out).expanduser()
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f'{out} exists; pass --overwrite to replace it')
        shutil.rmtree(out)
    out.mkdir(parents=True)

    trials = [json.loads(line) for line in (data / 'trials.jsonl').read_text().splitlines() if line]
    trials.sort(key=lambda t: (t['seed'], t['task_index']))
    if args.max_trials:
        trials = trials[:args.max_trials]

    datasets, logs, counts = {}, {}, defaultdict(int)
    infeasible = (out / 'infeasible.jsonl').open('w')
    images, image_seed = None, None
    try:
        for n, summary in enumerate(trials, 1):
            split = split_of(summary['seed'])
            if split not in datasets:
                datasets[split] = LeRobotDataset.create(
                    repo_id=f'{args.repo_prefix}_{split}', fps=args.fps, root=out / split,
                    features=features(args.height_map), robot_type='panda', use_videos=True)
                logs[split] = (out / f'{split}_episodes.jsonl').open('w')
            dataset = datasets[split]
            trial = json.loads((data / summary['trial']).read_text())
            scenario = json.loads((data / trial['scenario']).read_text())
            if image_seed != trial['seed']:
                images, image_seed = SceneImages(data, trial, args.height_map), trial['seed']

            for arm in args.arms:
                record = trial['arms'][arm]
                ref = dict(seed=trial['seed'], task_index=trial['task_index'], arm=arm,
                           cube=trial['cube'], tray=trial['tray'], trial=summary['trial'])
                if not record['feasible']:
                    infeasible.write(json.dumps(ref) + '\n')
                    continue
                plan = record['plans'][record['best_plan_index']]
                states = episode_states(plan, args.fps)
                env = environment_state(scenario, trial, arm)
                rgb = images.view(images.rgb, arm)
                height = images.view(images.height, arm) if args.height_map else None
                for i in range(len(states)):
                    frame = {'observation.state': states[i],
                             'observation.environment_state': env,
                             'observation.images.overhead': rgb,
                             'action': states[min(i + 1, len(states) - 1)],
                             'task': 'pick and place'}
                    if height is not None:
                        frame['observation.images.height'] = height
                    dataset.add_frame(frame)
                dataset.save_episode()
                logs[split].write(json.dumps(dict(
                    episode_index=counts[split], **ref, frames=len(states),
                    best_plan_index=record['best_plan_index'], cost=plan['cost'],
                    trajectory_duration=plan['total_trajectory_duration'],
                    joint_path_length=plan['total_joint_path_length'])) + '\n')
                counts[split] += 1
            if n % 100 == 0:
                print(f'{n}/{len(trials)} trials, episodes {dict(counts)}', flush=True)
    finally:
        infeasible.close()
        for log in logs.values():
            log.close()
        for dataset in datasets.values():
            dataset.finalize()

    (out / 'conversion.json').write_text(json.dumps(dict(
        source=str(data), trials=len(trials), episodes=dict(counts), fps=args.fps,
        arms=args.arms, splits={**SPLITS, 'train': 'all other seeds'},
        image_size=IMAGE_SIZE, second_arm_image_rotated_180=True, height_map=args.height_map,
        gripper=dict(open=GRIPPER_OPEN, closed=GRIPPER_CLOSE), bases=BASES,
        environment_state_names=ENV_NAMES,
        action='joint positions + gripper width at the next frame (absolute targets)'), indent=2))
    print(f'Done: {len(trials)} trials -> episodes {dict(counts)} in {out}')


if __name__ == '__main__':
    main()
