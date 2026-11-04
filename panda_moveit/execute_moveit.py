#!/usr/bin/env python3
"""MoveIt baseline in Gazebo, on the same trials as execute_policy.py.

For each trial (seed, task) of the plan-only data: apply the scenario, plan
the pick-and-place once with each arm (plan-only, timed: the decision), pick
the arm with the cheaper plan, then let that arm plan and execute it in
Gazebo the way collect_executions.py does. Success is the same test:
verify_place, the cube within 1.5 cm of the place target 1 s after release.

    python3 execute_moveit.py --seeds 1850 2000 --out ~/moveit_test.jsonl

Per trial: decision_time (both plans), each arm's plan cost, the chosen arm
and the labelled best arm (cheapest of the repeated plans in the data),
success and failed stage, motion time and cost of the executed motion,
place error, other cubes disturbed. Resumable: trials already in --out are
skipped.
"""
import argparse
import json
import math
import time
from pathlib import Path
from threading import Thread

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

# gz.msgs types come from pick_place, which imports them in a safe order.
from pick_place import ARMS, gz_model_poses, run, summarize
from collect import LIFT_HEIGHT, PREGRASP_HEIGHT, plan_arm
from scenario import ScenarioApplier

DISTURB = 0.01  # m, as in execute_policy.py


def run_trial(applier, trial, scenario):
    cube = next(c for c in scenario['cubes'] if c['name'] == trial['cube'])
    place_xyz = trial['place_xyz']
    applier.apply(scenario)

    # Decision: one plan per arm, as a planner-based assignment would do.
    start = time.monotonic()
    plans = {a: plan_arm(applier.arms[a], cube, place_xyz, repeats=1) for a in ARMS}
    decision_time = time.monotonic() - start
    costs = {a: p['best_cost'] for a, p in plans.items() if p['feasible']}
    arm = min(costs, key=costs.get) if costs else None
    result = dict(arm=arm, decision_time=decision_time,
                  plan_costs={a: p['best_cost'] for a, p in plans.items()},
                  plan_durations={a: p['best_trajectory_duration'] for a, p in plans.items()},
                  success=False, failed_stage=None if arm else 'no_plan')
    if arm is None:
        return result

    # Plan-only runs leave the scene as they found it, so execute from here.
    before = gz_model_poses()
    xyz, yaw = before[cube['name']]
    start = time.monotonic()
    stages = run(applier.arms[arm], cube['name'], xyz, place_xyz, PREGRASP_HEIGHT, LIFT_HEIGHT,
                 True, yaw, quiet=True)
    executed = summarize(stages, True)
    after = gz_model_poses()
    place = next((s for s in stages if s['stage'] == 'verify_place'), None)
    result.update(success=executed['success'], failed_stage=executed['failed_stage'],
                  execute_wall_time=time.monotonic() - start,
                  motion_time=executed['total_trajectory_duration'],
                  execute_planning_time=executed['total_planning_time'],
                  executed_cost=executed['cost'],
                  place_error=place['place_error'] if place else None,
                  disturbed=[c['name'] for c in scenario['cubes'] if c['name'] != cube['name']
                             and math.dist(after[c['name']][0], before[c['name']][0]) > DISTURB])
    return result


def summarize_rows(rows):
    n = len(rows)
    executed = [r for r in rows if r['arm']]
    m = max(len(executed), 1)
    print(f'\n== {n} trials, {len(executed)} executed')
    print(f"no arm planned: {sum(r['arm'] is None for r in rows)} "
          f"(labels have one in {sum(r['arm'] is None and r['label_arm'] is not None for r in rows)})")
    ok = sum(r['success'] for r in executed)
    print(f'task completed (cube within 1.5 cm): {ok}/{len(executed)} ({100 * ok / m:.1f}%)')
    stages = {}
    for r in executed:
        if r['failed_stage']:
            stages[r['failed_stage']] = stages.get(r['failed_stage'], 0) + 1
    print(f'failed at: {stages}')
    agree = sum(r['arm'] == r['label_arm'] for r in rows)
    print(f'same arm as the labels (best of repeated plans): {agree}/{n} ({100 * agree / n:.1f}%)')
    print(f"decision time median {np.median([r['decision_time'] for r in rows]):.3f} s")
    done = [r for r in executed if r.get('motion_time') is not None]
    if done:
        print(f"motion time median {np.median([r['motion_time'] for r in done]):.3f} s, "
              f"execution wall time median {np.median([r['execute_wall_time'] for r in done]):.2f} s")
    print(f"other cubes disturbed: {sum(bool(r.get('disturbed')) for r in executed)}/{len(executed)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs=2, default=(1850, 2000), metavar=('FIRST', 'END'))
    p.add_argument('--data', default=str(Path.home() / 'panda_data'))
    p.add_argument('--out', default=str(Path.home() / 'moveit_results.jsonl'))
    p.add_argument('--max-trials', type=int)
    p.add_argument('--speed', type=float, default=0.5)
    p.add_argument('--accel', type=float, default=0.5)
    args = p.parse_args()

    data = Path(args.data).expanduser()
    trials = [json.loads(line) for line in (data / 'trials.jsonl').read_text().splitlines() if line]
    trials = sorted((t for t in trials if args.seeds[0] <= t['seed'] < args.seeds[1]),
                    key=lambda t: (t['seed'], t['task_index']))
    trials = trials[:args.max_trials] if args.max_trials else trials
    out_path = Path(args.out).expanduser()
    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line] if out_path.exists() else []
    done = {(r['seed'], r['task_index']) for r in rows}

    rclpy.init()
    applier = ScenarioApplier(args.speed, args.accel)
    executor = SingleThreadedExecutor()
    for node in applier.arms.values():
        executor.add_node(node)
    spin = Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        with out_path.open('a') as out:
            for summary in trials:
                if (summary['seed'], summary['task_index']) in done:
                    continue
                trial = json.loads((data / summary['trial']).read_text())
                scenario = json.loads((data / trial['scenario']).read_text())
                labels = {a: v['best_cost'] for a, v in trial['arms'].items() if v['feasible']}
                row = dict(seed=trial['seed'], task_index=trial['task_index'],
                           label_arm=min(labels, key=labels.get) if labels else None)
                try:
                    row.update(run_trial(applier, trial, scenario))
                except Exception as e:  # unattended: record it and go on
                    row.update(arm=None, success=False, failed_stage='error', decision_time=None,
                               error=f'{type(e).__name__}: {e}')
                print(f"seed {row['seed']} task {row['task_index']} {row['arm']}: "
                      f"{'ok' if row['success'] else 'failed at ' + str(row['failed_stage'])}, "
                      f"decision {row['decision_time'] or 0:.2f} s", flush=True)
                out.write(json.dumps(row) + '\n')
                out.flush()
                rows.append(row)
    finally:
        executor.shutdown()
        for node in applier.arms.values():
            node.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=2)
    summarize_rows([r for r in rows if r['decision_time'] is not None] or rows)


if __name__ == '__main__':
    main()
