#!/usr/bin/env python3
"""Paired-trial data collection for the arm-assignment cost model.

For each seed: generate and apply a scenario, capture one RGB-D frame of the
initial scene, then for a few tasks (cube -> tray) evaluate both arms on the
identical scene and log them together as one paired trial.

Plan-only by default: each arm plans the full pick-and-place `--repeats`
times (OMPL is random, so a single plan is a noisy cost label). Every
attempt is logged with its joint trajectories; the cheapest feasible one
(cost T + lambda*L) is the arm's label and its policy training episode.
--execute also re-applies the scenario and executes the task once per arm,
logging the executed outcome next to the plans.

    collect.py --seeds 0 50
    collect.py --seeds 0 5 --execute --out ~/panda_data_exec

Output (--out, default ~/panda_data), resumable by seed:
    camera.json                    intrinsics and world pose of the camera
    scenarios/scenario_<seed>.json full scene, incl. arm start joints
    images/scene_<seed>_rgb.png    RGB before any trial
    images/scene_<seed>_depth.npz  float32 depth in metres (key 'depth'), same frame
    trials/seed_<seed>_task_<i>.json  both arms' plans with joint trajectories
    trials.jsonl                   one summary line per (seed, task)
"""
import argparse
import json
import math
import random
import time
from pathlib import Path
from threading import Event, Thread

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

# gz.msgs types come from pick_place, which imports them in a safe order.
from pick_place import (ARMS, COST_DEFINITION, PLANNER_SETTINGS, GzNode, TCP_OFFSET,
                        WORLD_FILE, gz_model_poses, quat_mul, run, summarize)
from gz.msgs10.camera_info_pb2 import CameraInfo
from gz.msgs10.image_pb2 import Image
from planning_scene_setup import parse_pose
from scenario import CUBE_Z, ScenarioApplier, generate

CAMERA_TOPIC = '/overhead_camera'
CAMERA_MODEL = 'overhead_rgbd'
# Gazebo cameras look along +x of their link (y left, z up); the usual
# optical frame looks along +z (x right, y down).
LINK_TO_OPTICAL = (-0.5, 0.5, -0.5, 0.5)  # xyzw
PREGRASP_HEIGHT = 0.15
LIFT_HEIGHT = 0.10


def capture(timeout=5.0):
    """One RGB frame, one depth frame and camera info, all received fresh."""
    got = {}
    done = Event()
    skip = {'rgb': 1, 'depth': 1}  # drop a frame possibly rendered mid-apply

    def keep(key, value):
        if key not in got:
            got[key] = value
            if len(got) == 3:
                done.set()

    def on_rgb(msg):
        if skip['rgb']:
            skip['rgb'] -= 1
            return
        keep('rgb', np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy())

    def on_depth(msg):
        if skip['depth']:
            skip['depth'] -= 1
            return
        keep('depth', np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width).copy())

    def on_info(msg):
        keep('info', dict(width=msg.width, height=msg.height, K=list(msg.intrinsics.k),
                          distortion=list(msg.distortion.k)))

    node = GzNode()
    node.subscribe(Image, f'{CAMERA_TOPIC}/image', on_rgb)
    node.subscribe(Image, f'{CAMERA_TOPIC}/depth_image', on_depth)
    node.subscribe(CameraInfo, f'{CAMERA_TOPIC}/camera_info', on_info)
    try:
        if not done.wait(timeout):
            raise TimeoutError(f'camera topics incomplete: got {sorted(got)}')
    finally:
        for topic in ('image', 'depth_image', 'camera_info'):
            node.unsubscribe(f'{CAMERA_TOPIC}/{topic}')
    return got['rgb'], got['depth'], got['info']


def save_depth(path, depth):
    """Lossless, compressed: float32 depth in metres under key 'depth' (~4x smaller)."""
    np.savez_compressed(path, depth=depth)


def camera_pose():
    """World pose of the camera link and optical frame, from the world SDF."""
    import xml.etree.ElementTree as ET
    world = ET.parse(WORLD_FILE).getroot().find('world')
    model = next(m for m in world.findall('model') if m.get('name') == CAMERA_MODEL)
    xyz, q = parse_pose(model.find('pose'))  # the sensor sits at the link origin
    return dict(position=list(xyz), link_quaternion_xyzw=list(q),
                optical_quaternion_xyzw=list(quat_mul(q, LINK_TO_OPTICAL)))


def choose_tasks(scenario, count):
    rng = random.Random(f"{scenario['seed']}-tasks")
    cubes = rng.sample(scenario['cubes'], count)
    return [dict(cube=c['name'], tray=rng.choice(scenario['trays'])['name']) for c in cubes]


def plan_arm(node, cube, place_xyz, repeats):
    """Plan-only attempts for one arm; the cheapest feasible one is its label."""
    plans = []
    for _ in range(repeats):
        start = time.monotonic()
        stages = run(node, cube['name'], cube['xyz'], place_xyz, PREGRASP_HEIGHT,
                     LIFT_HEIGHT, False, cube['yaw'], quiet=True, keep_trajectories=True)
        result = summarize(stages, False)
        result['wall_time'] = time.monotonic() - start
        result['hand_yaw'] = next((s['hand_yaw'] for s in stages if s['stage'] == 'grasp_select'
                                   and s['success']), None)
        result['stages'] = stages
        plans.append(result)
        if result['failed_stage'] == 'grasp_select':
            break  # no grasp yaw is reachable; replanning will not change that
    feasible = [p for p in plans if p['success']]
    best = min(feasible, key=lambda p: p['cost'], default=None)
    return dict(feasible=bool(feasible), plans=plans,
                best_plan_index=None if best is None else plans.index(best),
                best_cost=best and best['cost'],
                best_trajectory_duration=best and best['total_trajectory_duration'],
                best_joint_path_length=best and best['total_joint_path_length'],
                # What a planner-based assignment pays: one plan per arm.
                single_plan_wall_time=plans[0]['wall_time'],
                total_planning_wall_time=sum(p['wall_time'] for p in plans))


def evaluate(node, scenario, task, repeats, execute, applier):
    """Plan (and optionally execute) one task with one arm."""
    cube = next(c for c in scenario['cubes'] if c['name'] == task['cube'])
    tray = next(t for t in scenario['trays'] if t['name'] == task['tray'])
    place_xyz = (tray['xyz'][0], tray['xyz'][1], CUBE_Z)
    if execute:
        # The previous executed run moved a cube and both arms. Plan-only runs
        # change nothing, so plans and the execution below share this state.
        applier.apply(scenario)

    record = plan_arm(node, cube, place_xyz, repeats)
    if execute:
        live = gz_model_poses()[cube['name']]
        stages = run(node, cube['name'], live[0], place_xyz, PREGRASP_HEIGHT,
                     LIFT_HEIGHT, True, live[1], quiet=True, keep_trajectories=True)
        record['executed'] = dict(**summarize(stages, True), stages=stages)
    return record


def task_poses(scenario, task):
    cube = next(c for c in scenario['cubes'] if c['name'] == task['cube'])
    tray = next(t for t in scenario['trays'] if t['name'] == task['tray'])
    return dict(cube_pose=dict(xyz=cube['xyz'], yaw=cube['yaw']),
                dropoff_pose=dict(xyz=tray['xyz'], yaw=tray['yaw']),
                place_xyz=[tray['xyz'][0], tray['xyz'][1], CUBE_Z])


ARM_SUMMARY = ('feasible', 'best_plan_index', 'best_cost', 'best_trajectory_duration',
               'best_joint_path_length', 'single_plan_wall_time', 'total_planning_wall_time')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs=2, metavar=('FIRST', 'COUNT'), required=True)
    p.add_argument('--tasks', type=int, default=4, help='tasks per scenario')
    p.add_argument('--repeats', type=int, default=3, help='plans per arm per task')
    p.add_argument('--execute', action='store_true', help='also execute each task once per arm')
    p.add_argument('--out', default=str(Path.home() / 'panda_data'))
    p.add_argument('--speed', type=float, default=0.5)
    p.add_argument('--accel', type=float, default=0.5)
    args = p.parse_args()

    out = Path(args.out).expanduser()
    (out / 'scenarios').mkdir(parents=True, exist_ok=True)
    (out / 'images').mkdir(exist_ok=True)
    (out / 'trials').mkdir(exist_ok=True)
    trials_path = out / 'trials.jsonl'
    done = set()
    if trials_path.exists():
        done = {json.loads(line)['seed'] for line in trials_path.read_text().splitlines() if line}

    rclpy.init()
    applier = ScenarioApplier(args.speed, args.accel)
    executor = SingleThreadedExecutor()
    for node in applier.arms.values():
        executor.add_node(node)
    spin = Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        for seed in range(args.seeds[0], args.seeds[0] + args.seeds[1]):
            if seed in done:
                print(f'seed {seed}: already collected, skipping')
                continue
            start = time.monotonic()
            scenario = applier.apply(generate(seed))
            rgb, depth, info = capture()
            if not (out / 'camera.json').exists():
                (out / 'camera.json').write_text(json.dumps(dict(**info, **camera_pose()), indent=2))
            (out / 'scenarios' / f'scenario_{seed}.json').write_text(json.dumps(scenario, indent=2))
            cv2.imwrite(str(out / 'images' / f'scene_{seed}_rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            save_depth(out / 'images' / f'scene_{seed}_depth.npz', depth)

            lines = []
            for i, task in enumerate(choose_tasks(scenario, args.tasks)):
                arms = {name: evaluate(applier.arms[name], scenario, task, args.repeats,
                                       args.execute, applier)
                        for name in ARMS}
                trial_file = f'trials/seed_{seed}_task_{i}.json'
                info = dict(seed=seed, task_index=i, **task, **task_poses(scenario, task),
                            speed=args.speed, accel=args.accel, tcp_offset=TCP_OFFSET,
                            cost_definition=COST_DEFINITION, repeats=args.repeats,
                            planner=PLANNER_SETTINGS,
                            rgb=f'images/scene_{seed}_rgb.png',
                            depth=f'images/scene_{seed}_depth.npz',
                            scenario=f'scenarios/scenario_{seed}.json')
                # Full plans (with trajectories) per trial; the summary line stays small.
                (out / trial_file).write_text(json.dumps(dict(**info, arms=arms)) + '\n')
                lines.append(dict(**info, trial=trial_file,
                                  arms={n: {k: a[k] for k in ARM_SUMMARY} for n, a in arms.items()}))
                costs = {n: ('%.2f' % a['best_cost']) if a['feasible'] else 'infeasible'
                         for n, a in arms.items()}
                print(f"seed {seed} task {i}: {task['cube']} -> {task['tray']}: {costs}")
            # Whole scenarios only, so a resumed run never sees half a seed.
            with trials_path.open('a') as f:
                for line in lines:
                    f.write(json.dumps(line) + '\n')
            print(f'seed {seed}: {len(lines)} paired trials in {time.monotonic() - start:.0f}s')
    finally:
        executor.shutdown()
        for node in applier.arms.values():
            node.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=2)


if __name__ == '__main__':
    main()
