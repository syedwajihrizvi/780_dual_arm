#!/usr/bin/env python3
"""Execute-only demonstrations for closed-loop policies (e.g. ACT).

For each seed: apply the scenario, then for each task (cube -> tray) let
each arm plan and execute the pick-and-place in Gazebo while recorder.py
films it: RGB video, depth, both arms' joint states, gripper commands and
the cube's pose at 10 Hz. The scene is re-applied before an arm runs if the
previous run moved anything. No repeated plan-only attempts, so it is about
twice as fast as collect.py --execute, but there are no cost labels.

    collect_executions.py --seeds 6000 500 --out ~/panda_act

Output (--out), resumable by seed:
    camera.json
    scenarios/scenario_<seed>.json
    images/scene_<seed>_rgb.png, scene_<seed>_depth.npz   full-resolution before frame
    episodes/seed_<seed>_task_<i>_<arm>/
        rgb.mp4, depth.npz, frames.npz   the recording (see recorder.py)
        episode.json                     outcome, stages (t_end = s into the recording),
                                         executed joint trajectories
    episodes.jsonl                       one summary line per (seed, task, arm)
"""
import argparse
import json
import time
from pathlib import Path
from threading import Thread

import cv2
import rclpy
from rclpy.executors import SingleThreadedExecutor

# gz.msgs types come from pick_place, which imports them in a safe order.
from pick_place import ARMS, TCP_OFFSET, gz_model_poses, run, summarize
from collect import (LIFT_HEIGHT, MAX_FAILURES, PREGRASP_HEIGHT, SEED_ATTEMPTS, camera_pose,
                     capture, choose_tasks, save_depth, task_poses)
from recorder import Recorder
from scenario import ScenarioApplier, generate


def execute(node, recorder, task, scenario, folder):
    """Plan and execute one task with one arm while recording it."""
    cube = task['cube']
    place_xyz = task_poses(scenario, task)['place_xyz']
    xyz, yaw = gz_model_poses()[cube]
    t0 = recorder.start(folder, cube)
    try:
        stages = run(node, cube, xyz, place_xyz, PREGRASP_HEIGHT, LIFT_HEIGHT, True, yaw,
                     quiet=True, keep_trajectories=True)
    finally:
        recording = recorder.stop()
    for stage in stages:
        stage['t_end'] -= t0  # seconds into the recording
    return dict(**summarize(stages, True), stages=stages, recording=recording)


def collect_seed(seed, applier, recorder, args, out):
    scenario = applier.apply(generate(seed))
    need_info = not (out / 'camera.json').exists()
    rgb, depth, info = capture(need_info=need_info)
    if need_info:
        (out / 'camera.json').write_text(json.dumps(dict(**info, **camera_pose()), indent=2))
    (out / 'scenarios' / f'scenario_{seed}.json').write_text(json.dumps(scenario, indent=2))
    cv2.imwrite(str(out / 'images' / f'scene_{seed}_rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    save_depth(out / 'images' / f'scene_{seed}_depth.npz', depth)

    lines, dirty = [], False
    for i, task in enumerate(choose_tasks(scenario, args.tasks)):
        for arm in args.arms:
            if dirty:
                applier.apply(scenario)  # put the cube and both arms back
            name = f'seed_{seed}_task_{i}_{arm}'
            folder = out / 'episodes' / name
            result = execute(applier.arms[arm], recorder, task, scenario, folder)
            # A run that failed grasp selection never moved the arm or the cube.
            dirty = result['failed_stage'] != 'grasp_select'
            info = dict(seed=seed, task_index=i, arm=arm, **task, **task_poses(scenario, task),
                        speed=args.speed, accel=args.accel, tcp_offset=TCP_OFFSET,
                        scenario=f'scenarios/scenario_{seed}.json',
                        rgb=f'images/scene_{seed}_rgb.png', depth=f'images/scene_{seed}_depth.npz')
            (folder / 'episode.json').write_text(json.dumps(dict(**info, **result)) + '\n')
            lines.append(dict(**info, folder=f'episodes/{name}', success=result['success'],
                              failed_stage=result['failed_stage'],
                              frames=result['recording']['frames'],
                              duration=result['recording'].get('duration')))
            outcome = 'ok' if result['success'] else f"failed at {result['failed_stage']}"
            print(f"seed {seed} task {i} {arm}: {task['cube']} -> {task['tray']}: {outcome}, "
                  f"{result['recording']['frames']} frames")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs=2, metavar=('FIRST', 'COUNT'), required=True)
    p.add_argument('--tasks', type=int, default=4, help='tasks per scenario')
    p.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    p.add_argument('--out', default=str(Path.home() / 'panda_act'))
    p.add_argument('--speed', type=float, default=0.5)
    p.add_argument('--accel', type=float, default=0.5)
    args = p.parse_args()

    out = Path(args.out).expanduser()
    for sub in ('scenarios', 'images', 'episodes'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    lines_path = out / 'episodes.jsonl'
    done = set()
    if lines_path.exists():
        done = {json.loads(line)['seed'] for line in lines_path.read_text().splitlines() if line}

    rclpy.init()
    applier = ScenarioApplier(args.speed, args.accel)
    executor = SingleThreadedExecutor()
    for node in applier.arms.values():
        executor.add_node(node)
    spin = Thread(target=executor.spin, daemon=True)
    spin.start()
    recorder = Recorder(applier.arms)
    failures = 0  # consecutive skipped seeds
    try:
        for seed in range(args.seeds[0], args.seeds[0] + args.seeds[1]):
            if seed in done:
                print(f'seed {seed}: already collected, skipping')
                continue
            start = time.monotonic()
            for attempt in range(SEED_ATTEMPTS):
                try:
                    lines = collect_seed(seed, applier, recorder, args, out)
                    break
                except Exception as e:  # unattended: a rerun retries skipped seeds
                    print(f'seed {seed}: attempt {attempt + 1} failed, {type(e).__name__}: {e}')
            else:
                print(f'seed {seed}: skipped')
                failures += 1
                if failures >= MAX_FAILURES:
                    raise RuntimeError(f'{failures} seeds in a row failed; check the sim')
                continue
            failures = 0
            with lines_path.open('a') as f:
                for line in lines:
                    f.write(json.dumps(line) + '\n')
            ok = sum(line['success'] for line in lines)
            print(f'seed {seed}: {ok}/{len(lines)} executions succeeded in {time.monotonic() - start:.0f}s')
    finally:
        recorder.close()
        executor.shutdown()
        for node in applier.arms.values():
            node.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=2)


if __name__ == '__main__':
    main()
