#!/usr/bin/env python3
"""Unattended demonstration runs: randomize, plan both arms, execute the best.

For each scenario:
  1. randomize the table (trays, 3-6 obstacles, cubes, arm starts) and
     capture the "before" RGB-D frame and joint states;
  2. pick a task (cube -> tray) at least one arm can do;
  3. plan the full pick-and-place with both arms (--repeats times each),
     recording cost and MoveIt planning time for every attempt - planning
     both arms is the no-model baseline for choosing an arm;
  4. execute the task with the cheaper arm, then capture the "after" frame,
     joint states and where every cube ended up;
  5. write everything to data/scenario_<n>_seed<seed>/record.json (plus the
     images) and a summary line to data/demonstrations.jsonl.

    demonstrations.py                 # 10 scenarios, random seeds
    demonstrations.py --count 3 --seed-start 100
"""
import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState

# gz.msgs types come from pick_place, which imports them in a safe order.
from pick_place import ARMS, TCP_OFFSET, gz_model_poses, run, summarize
from collect import LIFT_HEIGHT, PREGRASP_HEIGHT, camera_pose, capture
from scenario import CUBE_Z, ScenarioApplier, generate

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
# Task cost C = T_total + LAMBDA * L_total (seconds + seconds per radian).
LAMBDA = 1.0
TASK_ATTEMPTS = 6  # tasks tried per scenario before giving up on it


class JointStates:
    """Latest /dual_joint_states (both arms, MoveIt-prefixed names)."""

    def __init__(self, node):
        self.msg = None
        node.create_subscription(JointState, '/dual_joint_states', self.on_msg, 10)

    def on_msg(self, msg):
        self.msg = msg

    def snapshot(self):
        if self.msg is None:
            return None
        return dict(zip(self.msg.name, self.msg.position))


def cost(result):
    return result['total_trajectory_duration'] + LAMBDA * result['total_joint_path_length']


def plan_arm(node, cube, place_xyz, repeats):
    """Plan-only attempts for one arm; the best feasible one sets its cost."""
    plans = []
    for _ in range(repeats):
        start = time.monotonic()
        stages = run(node, cube['name'], cube['xyz'], place_xyz, PREGRASP_HEIGHT,
                     LIFT_HEIGHT, False, cube['yaw'], quiet=True, keep_trajectories=True)
        result = summarize(stages, False)
        result['wall_time'] = time.monotonic() - start
        result['cost'] = cost(result) if result['success'] else None
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


def save_frame(folder, name):
    rgb, depth, _ = capture()
    cv2.imwrite(str(folder / f'{name}_rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(folder / f'{name}_depth.npy', depth)
    return dict(rgb=f'{name}_rgb.png', depth=f'{name}_depth.npy')


def pose_dict(poses, name):
    xyz, yaw = poses[name]
    return dict(xyz=list(xyz), yaw=yaw)


def run_scenario(index, seed, applier, joints, args, folder):
    record = dict(scenario_index=index, seed=seed, started=datetime.now().isoformat(),
                  speed=args.speed, accel=args.accel, tcp_offset=TCP_OFFSET,
                  repeats=args.repeats, cost_definition=f'T_total + {LAMBDA} * L_total')

    scenario = applier.apply(generate(seed))
    record['scenario'] = scenario
    record['images'] = dict(before=save_frame(folder, 'before'))
    record['joint_states'] = dict(before=joints.snapshot())
    before = gz_model_poses()
    record['cube_poses_before'] = {c['name']: pose_dict(before, c['name']) for c in scenario['cubes']}

    # A task at least one arm can complete.
    rng = random.Random(f'{seed}-task')
    candidates = rng.sample(scenario['cubes'], TASK_ATTEMPTS)
    record['rejected_tasks'] = []
    for cube in candidates:
        tray = rng.choice(scenario['trays'])
        place_xyz = (tray['xyz'][0], tray['xyz'][1], CUBE_Z)
        decide_start = time.monotonic()
        arms = {name: plan_arm(applier.arms[name], cube, place_xyz, args.repeats)
                for name in ARMS}
        decide_time = time.monotonic() - decide_start
        if any(a['feasible'] for a in arms.values()):
            break
        record['rejected_tasks'].append(dict(cube=cube['name'], tray=tray['name'],
                                             reason='infeasible for both arms'))
    else:
        raise RuntimeError(f'no feasible task in {TASK_ATTEMPTS} tries')

    record['task'] = dict(cube=cube['name'], cube_pose=dict(xyz=cube['xyz'], yaw=cube['yaw']),
                          dropoff=tray['name'], dropoff_pose=dict(xyz=tray['xyz'], yaw=tray['yaw']),
                          place_xyz=list(place_xyz))
    record['arms'] = arms
    assigned = min((n for n in ARMS if arms[n]['feasible']), key=lambda n: arms[n]['best_cost'])
    record['planner_assignment'] = dict(
        assigned_arm=assigned, rule='lowest best planned cost among feasible arms',
        costs={n: arms[n]['best_cost'] for n in ARMS},
        # Baseline decision time without a model: plan both arms once each.
        baseline_decision_time=sum(arms[n]['single_plan_wall_time'] for n in ARMS),
        decision_time_with_repeats=decide_time)

    # Execute with the assigned arm; plan-only runs left the scene unchanged.
    stages = run(applier.arms[assigned], cube['name'], cube['xyz'], place_xyz,
                 PREGRASP_HEIGHT, LIFT_HEIGHT, True, cube['yaw'], quiet=True,
                 keep_trajectories=True)
    executed = summarize(stages, True)
    executed['cost'] = cost(executed) if executed['success'] else None
    after = gz_model_poses()
    final = after[cube['name']][0]
    executed.update(
        arm=assigned, stages=stages, cube_final_pose=pose_dict(after, cube['name']),
        place_error=math.dist(final, place_xyz),
        # Did the run disturb anything else on the table?
        other_cubes_max_displacement=max(
            math.dist(after[c['name']][0], before[c['name']][0])
            for c in scenario['cubes'] if c['name'] != cube['name']))
    record['execution'] = executed
    record['completed_by'] = assigned if executed['success'] else None

    # The retreat ends right above the tray; move back to the start pose so
    # the after frame shows the placed cube instead of the arm.
    record['return_to_start'] = applier.arms[assigned].move_to_joints(
        scenario['arm_starts'][assigned])
    record['images']['after'] = save_frame(folder, 'after')
    record['joint_states']['after'] = joints.snapshot()
    record['cube_poses_after'] = {c['name']: pose_dict(after, c['name']) for c in scenario['cubes']}
    record['finished'] = datetime.now().isoformat()
    return record


def summary_line(record, folder):
    """Compact line for demonstrations.jsonl; the full record is in the folder."""
    line = dict(scenario_index=record['scenario_index'], seed=record['seed'],
                folder=folder.name, error=record.get('error'))
    if 'execution' in record:
        arms = record['arms']
        line.update(cube=record['task']['cube'], dropoff=record['task']['dropoff'],
                    obstacles=len(record['scenario']['obstacles']),
                    assigned_arm=record['planner_assignment']['assigned_arm'],
                    completed_by=record['completed_by'],
                    cost={n: arms[n]['best_cost'] for n in arms},
                    feasible={n: arms[n]['feasible'] for n in arms},
                    baseline_decision_time=record['planner_assignment']['baseline_decision_time'],
                    executed_success=record['execution']['success'],
                    executed_cost=record['execution']['cost'],
                    place_error=record['execution']['place_error'])
    return line


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=10)
    p.add_argument('--seed-start', type=int, help='default: random')
    p.add_argument('--repeats', type=int, default=3, help='plans per arm per task')
    p.add_argument('--out', default=str(DATA_DIR))
    p.add_argument('--speed', type=float, default=0.5)
    p.add_argument('--accel', type=float, default=0.5)
    args = p.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    seed_start = args.seed_start if args.seed_start is not None else random.randrange(10**6)

    rclpy.init()
    applier = ScenarioApplier(args.speed, args.accel)
    joints = JointStates(applier.arms['first'])
    executor = SingleThreadedExecutor()
    for node in applier.arms.values():
        executor.add_node(node)
    spin = Thread(target=executor.spin, daemon=True)
    spin.start()

    completed = 0
    try:
        if not (out / 'camera.json').exists():
            _, _, info = capture()
            (out / 'camera.json').write_text(json.dumps(dict(**info, **camera_pose()), indent=2))
        for index in range(1, args.count + 1):
            seed = seed_start + index - 1
            folder = out / f'scenario_{index:03d}_seed{seed}'
            folder.mkdir(exist_ok=True)
            print('Randomizing pieces', flush=True)
            print(f'Running Scenario {index}...', flush=True)
            try:
                record = run_scenario(index, seed, applier, joints, args, folder)
                done = record['execution']['success']
                completed += done
                costs = ', '.join(f"{n} {c:.2f}" if c is not None else f'{n} infeasible'
                                  for n, c in record['planner_assignment']['costs'].items())
                print(f'Completed Scenario {index}...', flush=True)
                print(f"    {record['task']['cube']} -> {record['task']['dropoff']}, "
                      f"{len(record['scenario']['obstacles'])} obstacles | cost {costs} | "
                      f"{record['planner_assignment']['assigned_arm']} arm "
                      f"{'completed it' if done else 'FAILED at ' + str(record['execution']['failed_stage'])}",
                      flush=True)
            except Exception as e:  # keep going: the run must be unattended
                record = dict(scenario_index=index, seed=seed, error=f'{type(e).__name__}: {e}')
                print(f'Scenario {index} failed: {record["error"]}', flush=True)
            (folder / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
            with (out / 'demonstrations.jsonl').open('a') as f:
                f.write(json.dumps(summary_line(record, folder)) + '\n')
            print(f'Write data for scenario {index} to file for demonstration', flush=True)
        print(f'Done: {completed}/{args.count} scenarios completed. Data in {out}', flush=True)
    finally:
        executor.shutdown()
        for node in applier.arms.values():
            node.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=2)


if __name__ == '__main__':
    main()
