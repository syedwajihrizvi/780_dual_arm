#!/usr/bin/env python3
"""Convert collect_executions.py recordings to LeRobot datasets for
closed-loop policies (ACT).

One episode = one successful recorded execution. Unlike convert_lerobot.py
(planned trajectories, the before image on every frame), every frame here
has the live camera view and the measured joint state:
    observation.state              7 measured joints + gripper width (mean finger position)
    observation.environment_state  task and scene in the arm's base frame, as convert_lerobot.py
    observation.images.overhead    live RGB (video)
    observation.images.height      live height map from depth (--height-map; images, not video)
    action                         next frame's measured joints + the commanded gripper width
Both arms share the datasets the same way: arm base frame, image rotated
180 degrees for the second arm, arm ID in the environment state. The idle
second at the end of each recording (the cube settling) is trimmed.

    ~/lerobot_venv/bin/python convert_executions.py --data ~/panda_act --out ~/lerobot_act --height-map

Writes <out>/train and <out>/val (seeds ending in 9), <split>_episodes.jsonl
(episode index -> seed, task, arm, source folder) and conversion.json.
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from convert_lerobot import (CAMERA_Z, ENV_NAMES, GRIPPER_OPEN, IMAGE_SIZE, MAX_HEIGHT, TABLE_Z,
                             environment_state, features)

RECORD_FPS = 10   # recorder.FPS
STILL = 0.002     # rad per frame; a joint moving less is still
KEEP_END = 2      # still frames kept after the motion ends


def split_of(seed):
    return 'val' if seed % 10 == 9 else 'train'


def read_video(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            return frames
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def height_map(depth_mm):
    """uint16 depth in mm (0 = no return) -> 3-channel uint8 height above
    the table, scaled like convert_lerobot.SceneImages."""
    depth = depth_mm.astype(np.float32) / 1000.0
    depth[depth_mm == 0] = np.inf  # background reads as table height 0
    h = np.clip((CAMERA_Z - TABLE_Z) - depth, 0.0, MAX_HEIGHT)
    h = cv2.resize((h / MAX_HEIGHT * 255).astype(np.uint8), IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return np.repeat(h[:, :, None], 3, axis=2)


def view(image, arm):
    # Same as convert_lerobot.SceneImages.view: each arm sees the table from its side.
    return image if arm == 'first' else cv2.rotate(image, cv2.ROTATE_180)


def episode_frames(frames, arm, step):
    """States (n, 8), actions (n, 8) and the recording rows they come from."""
    q = frames[f'{arm}_position']
    states = np.hstack([q[:, :7], q[:, 7:9].mean(axis=1, keepdims=True)])
    command = frames[f'{arm}_gripper_command'].copy()
    command[np.isnan(command)] = GRIPPER_OPEN  # before the first command: open
    # Trim the idle tail: keep KEEP_END frames after the last change.
    changes = (np.abs(np.diff(q[:, :7], axis=0)).max(axis=1) > STILL) | (np.diff(command) != 0)
    moving = np.flatnonzero(changes)
    end = min(len(q), (moving[-1] + 2 if len(moving) else 1) + KEEP_END)
    rows = np.arange(0, end, step)
    nxt = np.minimum(rows + step, end - 1)
    actions = np.hstack([q[nxt, :7], command[nxt, None]])
    return states[rows].astype(np.float32), actions.astype(np.float32), rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(Path.home() / 'panda_act'))
    p.add_argument('--out', default=str(Path.home() / 'lerobot_act'))
    p.add_argument('--repo-prefix', default='dual_panda_act', help='repo_id is <prefix>_<split>')
    p.add_argument('--fps', type=int, default=RECORD_FPS, choices=(10, 5))
    p.add_argument('--arms', nargs='+', default=['first', 'second'])
    p.add_argument('--height-map', action='store_true', help='add the live depth height map')
    p.add_argument('--max-episodes', type=int, help='convert only the first N (testing)')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    data, out = Path(args.data).expanduser(), Path(args.out).expanduser()
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f'{out} exists; pass --overwrite to replace it')
        shutil.rmtree(out)
    out.mkdir(parents=True)

    lines = [json.loads(line) for line in (data / 'episodes.jsonl').read_text().splitlines() if line]
    lines = sorted((l for l in lines if l['success'] and l['arm'] in args.arms),
                   key=lambda l: (l['seed'], l['task_index'], l['arm']))
    if args.max_episodes:
        lines = lines[:args.max_episodes]
    step = RECORD_FPS // args.fps

    datasets, logs, counts, skipped = {}, {}, defaultdict(int), []
    scenarios = {}
    try:
        for n, line in enumerate(lines, 1):
            folder, arm = data / line['folder'], line['arm']
            frames = np.load(folder / 'frames.npz')
            rgb = read_video(folder / 'rgb.mp4')
            depth = np.load(folder / 'depth.npz')['depth_mm'] if args.height_map else None
            rows_recorded = len(frames['t'])
            if len(rgb) != rows_recorded or (depth is not None and len(depth) != rows_recorded):
                skipped.append(dict(folder=line['folder'], reason='frame counts differ'))
                continue

            split = split_of(line['seed'])
            if split not in datasets:
                datasets[split] = LeRobotDataset.create(
                    repo_id=f'{args.repo_prefix}_{split}', fps=args.fps, root=out / split,
                    features=features(args.height_map), robot_type='panda', use_videos=True)
                logs[split] = (out / f'{split}_episodes.jsonl').open('w')
            if line['seed'] not in scenarios:
                scenarios = {line['seed']: json.loads((data / line['scenario']).read_text())}
            env = environment_state(scenarios[line['seed']], line, arm)

            states, actions, rows = episode_frames(frames, arm, step)
            for i, row in enumerate(rows):
                image = cv2.resize(rgb[row], IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                frame = {'observation.state': states[i],
                         'observation.environment_state': env,
                         'observation.images.overhead': view(image, arm),
                         'action': actions[i],
                         'task': 'pick and place'}
                if depth is not None:
                    frame['observation.images.height'] = view(height_map(depth[row]), arm)
                datasets[split].add_frame(frame)
            datasets[split].save_episode()
            logs[split].write(json.dumps(dict(
                episode_index=counts[split], seed=line['seed'], task_index=line['task_index'],
                arm=arm, cube=line['cube'], tray=line['tray'], folder=line['folder'],
                frames=len(rows), recorded_frames=rows_recorded)) + '\n')
            counts[split] += 1
            if n % 50 == 0:
                print(f'{n}/{len(lines)} executions, episodes {dict(counts)}', flush=True)
    finally:
        for log in logs.values():
            log.close()
        for dataset in datasets.values():
            dataset.finalize()

    (out / 'conversion.json').write_text(json.dumps(dict(
        source=str(data), executions=len(lines), episodes=dict(counts), skipped=skipped,
        fps=args.fps, arms=args.arms, splits=dict(val='seeds ending in 9', train='all other seeds'),
        image_size=IMAGE_SIZE, second_arm_image_rotated_180=True, height_map=args.height_map,
        environment_state_names=ENV_NAMES, idle_end_frames_kept=KEEP_END,
        state='measured joints + mean finger position',
        action='next frame measured joints + commanded gripper width (absolute targets)'), indent=2))
    print(f'Done: {len(lines)} successful executions -> episodes {dict(counts)} in {out}'
          + (f', {len(skipped)} skipped' if skipped else ''))


if __name__ == '__main__':
    main()
