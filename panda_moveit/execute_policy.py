#!/usr/bin/env python3
"""Execute the policy's chosen trajectories in Gazebo.

Takes the rows evaluate_policy.py --save-trajectories wrote (same seeds and
data) and, for each trial where the policy chose an arm, replays the chosen
sample on that arm, sent straight to the arm controller (no MoveIt planning):
    frames up to the grasp -> close the gripper (it must stall on the cube) and attach
    -> frames up to the release -> detach and open -> the remaining frames
The scenario is re-applied before every trial.

Success is the test collect_executions.py uses: 1 s after the motion ends the
cube is within PLACE_TOLERANCE of the place target. Collisions, three ways:
    moveit_contacts  MoveIt's state check (full meshes, every obstacle, the
                     other arm) on the commanded motion, SUBSTEPS per frame.
                     Contacts with the task's cube are ignored (the fingers
                     touch it on purpose) and the carried cube is not checked.
    tracking_error   largest gap between commanded and measured joints while
                     moving, after allowing for the controller's lag (up to
                     MAX_LAG); obstacles are static in Gazebo, so an arm that
                     hits one is held back instead of moving it.
    disturbed        other cubes that moved more than 1 cm.

    ~/lerobot_venv/bin/python evaluate_policy.py --checkpoint <ckpt> --seeds 1700 1850 \\
        --save-trajectories --out ~/eval_exec.jsonl
    python3 execute_policy.py --plans ~/eval_exec.jsonl --out ~/exec_results.jsonl

Two options change what is executed (the plans need evaluate_policy.py's
candidates, from --save-trajectories):
    --exact-check  try the samples in evaluate_policy.py's order and execute
                   the first with no MoveIt contact; none clean -> no arm.
                   The check runs on a local copy of MoveIt's scene, and its
                   time is added to the decision time.
    --retime       skip the frames where the arm barely moves (samples often
                   idle for seconds after the release, jittering just enough
                   not to count as stopped); where it moves, keep the sampled
                   5 Hz timing (that of MoveIt's demonstrations), never above
                   the joint velocity limits scaled by --speed.

Resumable: trials already in --out are skipped.
"""
import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from threading import Lock, Thread

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from moveit.core.collision_detection import CollisionRequest, CollisionResult
from moveit.core.planning_scene import PlanningScene
from moveit.core.robot_model import RobotModel
from moveit.core.robot_state import RobotState
from moveit_msgs.msg import CollisionObject, PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene, GetStateValidity
from rcl_interfaces.srv import GetParameters
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

# gz.msgs types come from pick_place, which imports them in a safe order.
from pick_place import ARMS, GzGrip, gz_model_poses, wait_done
from panda_kinematics import tcp_position
from scenario import ScenarioApplier

GRIPPER_MID = (0.04 + 0.015) / 2  # between open and closed, as in evaluate_policy.py
PLACE_TOLERANCE = 0.015           # m; collect_executions.py's verify_place
TRACKING_LIMIT = 0.15             # rad; free motion stays under ~0.08 after the lag fit
DISTURB = 0.01                    # m
SUBSTEPS = 4
MAX_LAG = 0.5                     # s
GRASP_REACH = 0.03                # m; hand further than this from the cube cannot grasp it
IDLE = 0.01                       # rad; --retime merges frames until a joint moves this much


class JointLog:
    """Measured arm joints of one arm while recording."""

    def __init__(self, node):
        self.lock = Lock()
        self.samples = None
        node.create_subscription(JointState, f'{node.arm.ns}/joint_states', self.on_joints, 50)

    def on_joints(self, msg):
        with self.lock:
            if self.samples is None:
                return
            pos = dict(zip(msg.name, msg.position))
            if all(f'panda_joint{i}' in pos for i in range(1, 8)):
                self.samples.append((time.monotonic(), [pos[f'panda_joint{i}'] for i in range(1, 8)]))

    def start(self):
        with self.lock:
            self.samples = []

    def stop(self):
        with self.lock:
            samples, self.samples = self.samples, None
        return samples


def follow(node, log, frames, times):
    """Send frames[1:] to the arm controller at times[1:] (s), frames[0] being
    where the arm is. Returns the controller's error code, the largest tracking
    error and the last measured joints."""
    if len(frames) < 2:
        return dict(error_code=0, tracking_error=0.0, measured=list(frames[0][:7]))
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = [f'panda_joint{i}' for i in range(1, 8)]
    for q, t in zip(frames[1:], times[1:]):
        goal.trajectory.points.append(JointTrajectoryPoint(
            positions=[float(v) for v in q[:7]],
            time_from_start=Duration(sec=int(t), nanosec=int(t % 1 * 1e9))))
    if not node.controller.wait_for_server(timeout_sec=15.0):
        raise RuntimeError(f'{node.arm.name} arm controller unavailable')
    log.start()
    start = time.monotonic()
    handle = wait_done(node.controller.send_goal_async(goal), 20)
    if not handle.accepted:
        log.stop()
        raise RuntimeError('controller rejected the trajectory')
    result = wait_done(handle.get_result_async(), times[-1] + 20).result
    samples = log.stop()
    # Commanded joints at each measurement time (the controller interpolates
    # between points; linear is close enough at 5 Hz for a tolerance check).
    # The arm trails the command by ~0.1-0.3 s even when free, so compare with
    # the command at the lag that fits best; a blocked arm fits no lag.
    times = np.asarray(times)
    q = np.asarray(frames)[:, :7]
    t = np.array([s[0] - start for s in samples])
    measured = np.array([s[1] for s in samples]).reshape(-1, 7)
    error = 0.0
    if len(t):
        error = min(
            float(np.abs(measured - np.stack([np.interp(np.clip(t - lag, 0, times[-1]), times, q[:, j])
                                              for j in range(7)], axis=1)).max())
            for lag in np.arange(0, MAX_LAG + 1e-9, 0.05))
    return dict(error_code=int(result.error_code), tracking_error=error,
                measured=measured[-1].tolist() if len(t) else list(q[-1]))


def dense(frames):
    """The commanded motion, SUBSTEPS states per frame."""
    q = np.asarray(frames)
    return [q[0]] + [q[i - 1] + (q[i] - q[i - 1]) * s / SUBSTEPS
                     for i in range(1, len(q)) for s in range(1, SUBSTEPS + 1)]


class SceneChecker:
    """MoveIt's collision check (move_group's robot model, the world objects
    of its scene) run in this process: ~0.1 ms a state instead of a service
    round trip each. The Python bindings report whether a state collides but
    not with what, so the names come from move_group for the colliding states."""

    def __init__(self, applier):
        self.applier = applier
        node = next(iter(applier.arms.values()))
        req = GetParameters.Request()
        req.names = ['robot_description', 'robot_description_semantic']
        urdf, srdf = applier.call(node.create_client(GetParameters, '/move_group/get_parameters'), req).values
        folder = Path(tempfile.mkdtemp())
        (folder / 'robot.urdf').write_text(urdf.string_value)
        (folder / 'robot.srdf').write_text(srdf.string_value)
        self.model = RobotModel(str(folder / 'robot.urdf'), str(folder / 'robot.srdf'))
        self.scene = PlanningScene(self.model)
        self.request = CollisionRequest()
        self.request.joint_model_group_name = 'both_panda_arms'

    def load(self, cube, starts):
        """Copy move_group's current world, without the task's cube (the fingers
        touch it on purpose), with both arms at their start joints and open.
        (move_group's robot state comes back all zeros: the other arm would
        stand straight up.)"""
        req = GetPlanningScene.Request()
        req.components.components = (PlanningSceneComponents.WORLD_OBJECT_NAMES
                                      | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY)
        msg = self.applier.call(self.applier.get_scene, req).scene
        # Adding a world is additive and a new PlanningScene still held the
        # previous trial's obstacles, so clear it first.
        self.scene.remove_all_collision_objects()
        self.scene.process_planning_scene_world(msg.world)
        remove = CollisionObject(id=cube, operation=CollisionObject.REMOVE)
        remove.header.frame_id = self.scene.planning_frame
        self.scene.apply_collision_object(remove)
        self.state = RobotState(self.model)
        self.state.joint_positions = {
            name: value for arm, q in starts.items()
            for name, value in zip([f'{arm}_panda_joint{i}' for i in range(1, 8)]
                                   + [f'{arm}_panda_finger_joint1', f'{arm}_panda_finger_joint2'],
                                   list(q[:7]) + [0.04, 0.04])}
        self.state.update()

    def colliding(self, arm, state):
        width = float(min(max(state[7], 0.0), 0.04))
        self.state.joint_positions = dict(zip(
            [f'{arm}_panda_joint{i}' for i in range(1, 8)]
            + [f'{arm}_panda_finger_joint1', f'{arm}_panda_finger_joint2'],
            [float(v) for v in state[:7]] + [width, width]))
        self.state.update()
        result = CollisionResult()
        self.scene.check_collision(self.request, result, self.state, self.scene.allowed_collision_matrix)
        return result.collision

    def clear(self, arm, frames):
        return not any(self.colliding(arm, q) for q in dense(frames))

    def contacts(self, arm, frames, cube):
        """Objects in contact with the arm along the commanded motion."""
        hits = set()
        for state in dense(frames):
            if not self.colliding(arm, state):
                continue
            req = GetStateValidity.Request()
            req.group_name = 'both_panda_arms'
            req.robot_state.is_diff = True
            req.robot_state.joint_state.name = ([f'{arm}_panda_joint{i}' for i in range(1, 8)]
                                                + [f'{arm}_panda_finger_joint1'])
            req.robot_state.joint_state.position = [float(v) for v in state[:7]] + [
                float(min(max(state[7], 0.0), 0.04))]
            for c in self.applier.call(self.applier.validity, req).contacts:
                bodies = {c.contact_body_1, c.contact_body_2}
                if cube in bodies:
                    continue
                hits |= {b for b in bodies if not b.startswith(f'{arm}_panda_')}
        return sorted(hits)


def velocity_limits(speed):
    """Velocity limits of the 7 arm joints (rad/s), scaled as MoveIt scales them."""
    path = Path(get_package_share_directory('moveit_resources_panda_moveit_config')) / 'config' / 'joint_limits.yaml'
    limits = yaml.safe_load(path.read_text())['joint_limits']
    return speed * np.array([limits[f'panda_joint{i}']['max_velocity'] for i in range(1, 8)])


def retime(frames, vmax, fps):
    """Drop frames until some joint has moved IDLE from the last one kept, then
    time each step 1/fps, or longer if a joint would exceed vmax. The last
    frame (where the gripper acts) is kept. Returns (frames, times)."""
    q = np.asarray(frames)
    keep = [0]
    for i in range(1, len(q)):
        if np.abs(q[i, :7] - q[keep[-1], :7]).max() > IDLE:
            keep.append(i)
    if keep[-1] != len(q) - 1:
        if len(keep) > 1:
            keep[-1] = len(q) - 1
        else:
            keep.append(len(q) - 1)
    q = q[keep]
    dt = np.maximum(1 / fps, (np.abs(np.diff(q[:, :7], axis=0)) / vmax).max(axis=1))
    return q, np.concatenate([[0.0], np.cumsum(dt)])


def switch_frame(traj, start, closing):
    """First frame from start where the gripper closes (or opens)."""
    g = traj[start:, 7]
    idx = np.flatnonzero(g < GRIPPER_MID if closing else g > GRIPPER_MID)
    return start + int(idx[0]) if len(idx) else None


def closest(traj, arm, frame, target):
    """The frame before the switch or the switch frame, whichever is nearer the
    target (at 5 Hz the gripper acts between them; as in evaluate_policy.py)."""
    candidates = [max(frame - 1, 0), frame]
    tcp = tcp_position(traj[candidates, :7], arm)
    return candidates[int(np.argmin(np.linalg.norm(tcp - np.asarray(target), axis=1)))]


def execute(applier, checker, logs, plan, trial, scenario, args, vmax):
    cube = trial['cube']
    applier.apply(scenario)
    before = gz_model_poses()
    checker.load(cube, scenario['arm_starts'])
    candidates = plan.get('candidates') or [dict(arm=plan['policy_arm'], trajectory=plan['trajectory'],
                                                 score=plan['trajectory_score'])]
    result = dict(candidate=0, exact_check_time=None)
    if args.exact_check:
        # The first candidate (what runs without the check) is always checked,
        # so the report can say how often the check changed the choice.
        start = time.monotonic()
        clear = [checker.clear(c['arm'], c['trajectory']) for c in candidates[:1]]
        while not clear[-1] and len(clear) < len(candidates):
            clear.append(checker.clear(candidates[len(clear)]['arm'], candidates[len(clear)]['trajectory']))
        result.update(exact_check_time=time.monotonic() - start, candidates_checked=len(clear),
                      first_candidate_clear=clear[0])
        if not clear[-1]:
            result.update(arm=None, policy_arm=None, success=False, failed_stage='no_clear_candidate')
            return result
        result['candidate'] = len(clear) - 1
    chosen = candidates[result['candidate']]
    arm = chosen['arm']
    node = applier.arms[arm]
    traj = np.asarray(chosen['trajectory'])
    result.update(arm=arm, policy_arm=arm, frames=len(traj), success=False, failed_stage=None,
                  policy_cost=chosen['score']['cost'], sampled_duration=(len(traj) - 1) / args.fps)
    result['moveit_contacts'] = checker.contacts(arm, traj, cube)

    grasp = switch_frame(traj, 0, closing=True)
    release = switch_frame(traj, grasp, closing=False) if grasp is not None else None
    if release is None:  # evaluate_policy only picks samples that grasp and release
        result['failed_stage'] = 'no_grasp_or_release'
        return result
    grasp = closest(traj, arm, grasp, trial['cube_pose']['xyz'])
    release = max(closest(traj, arm, release, trial['place_xyz']), grasp)
    result.update(grasp_frame=grasp, release_frame=release)

    def timed(frames):
        if args.retime:
            return retime(frames, vmax, args.fps)
        return frames, np.arange(len(frames)) / args.fps
    parts = [timed(traj[:grasp + 1]), timed(traj[grasp:release + 1]), timed(traj[release:])]
    result['policy_duration'] = sum(float(t[-1]) for _, t in parts)

    segments = []
    t0 = time.monotonic()
    grip = GzGrip(node.arm, cube)
    try:
        segments.append(follow(node, logs[arm], *parts[0]))
        # Where the hand and the cube are when the fingers close: a cube pushed
        # or turned on the way down is grasped off-centre or across its corners.
        cube_xyz, cube_yaw = gz_model_poses()[cube]
        result['cube_pushed'] = math.dist(cube_xyz, before[cube][0])
        result['cube_turned'] = math.degrees(abs((cube_yaw - before[cube][1] + math.pi / 4) % (math.pi / 2)
                                                 - math.pi / 4))
        tcp = tcp_position(np.asarray(segments[-1]['measured']), arm)
        result['tcp_to_cube'] = [float(v) for v in np.subtract(tcp, cube_xyz)]
        # The attach joint holds the cube wherever it is, so check the hand
        # reached it: an arm stopped short (e.g. on an obstacle) would otherwise
        # carry the cube from a distance.
        if math.dist(tcp, cube_xyz) > GRASP_REACH:
            result['failed_stage'] = 'grasp_missed'
            return result
        held = node.close_gripper()
        result['grasp_gap'] = held['gap']
        if not held['success']:
            result['failed_stage'] = 'grasp'
            return result
        if not grip.attach()['success']:
            result['failed_stage'] = 'attach'
            return result
        segments.append(follow(node, logs[arm], *parts[1]))
        grip.detach()
        node.open_gripper()
        segments.append(follow(node, logs[arm], *parts[2]))
        time.sleep(1.0)  # let the cube settle
    finally:
        result['duration'] = time.monotonic() - t0
        result['tracking_error'] = max((s['tracking_error'] for s in segments), default=0.0)
        result['controller_errors'] = [s['error_code'] for s in segments if s['error_code']]
        after = gz_model_poses()
        result['disturbed'] = [c['name'] for c in scenario['cubes'] if c['name'] != cube
                               and math.dist(after[c['name']][0], before[c['name']][0]) > DISTURB]
        result['place_error'] = math.dist(after[cube][0], trial['place_xyz'])
    result['success'] = result['place_error'] < PLACE_TOLERANCE
    if not result['success']:
        result['failed_stage'] = 'place'
    return result


def summarize(rows):
    executed = [r for r in rows if r.get('arm')]
    n = len(rows)
    print(f'\n== {n} trials, {len(executed)} executed')
    print(f"no arm chosen: {sum(r['policy_arm'] is None for r in rows)} "
          f"(MoveIt found one in {sum(r['policy_arm'] is None and r['moveit_arm'] is not None for r in rows)})")
    if not executed:
        return
    m = len(executed)

    def pct(k):
        return f'{k}/{m} ({100 * k / m:.1f}%)'
    print(f"task completed (cube within {100 * PLACE_TOLERANCE:.1f} cm): {pct(sum(r['success'] for r in executed))}")
    stages = {}
    for r in executed:
        if r['failed_stage']:
            stages[r['failed_stage']] = stages.get(r['failed_stage'], 0) + 1
    print(f'failed at: {stages}')
    contact = [r for r in executed if r['moveit_contacts']]
    held_back = [r for r in executed if r.get('tracking_error', 0) > TRACKING_LIMIT]
    print(f'MoveIt contact along the commanded motion: {pct(len(contact))}')
    objects = {}
    for r in contact:
        for o in r['moveit_contacts']:
            objects[o] = objects.get(o, 0) + 1
    if objects:
        print('  objects: ' + ', '.join(f'{k} {v}' for k, v in sorted(objects.items(), key=lambda kv: -kv[1])))
    print(f'held back in Gazebo (tracking error > {TRACKING_LIMIT} rad): {pct(len(held_back))}')
    print(f"other cubes disturbed: {pct(sum(bool(r.get('disturbed')) for r in executed))}")
    any_collision = [r for r in executed if r['moveit_contacts'] or r.get('tracking_error', 0) > TRACKING_LIMIT
                     or r.get('disturbed')]
    print(f'any collision sign: {pct(len(any_collision))}')
    errors = [r['place_error'] for r in executed if r.get('place_error') is not None]
    if errors:
        print(f'place error median {100 * np.median(errors):.1f} cm')

    print('\n== policy vs MoveIt')
    agree = sum(r['policy_arm'] == r['moveit_arm'] for r in rows)
    print(f'same arm as MoveIt: {agree}/{n} ({100 * agree / n:.1f}%)')
    for name, pk, mk in (('decision time', 'policy_decision_time', 'moveit_decision_time'),
                         ('motion time', 'policy_duration', 'moveit_duration'),
                         ('cost T + L', 'policy_cost', 'moveit_cost')):
        both = [r for r in executed if r.get(pk) is not None and r.get(mk) is not None]
        if both:
            pol = np.array([r[pk] for r in both])
            mov = np.array([r[mk] for r in both])
            print(f'{name}: policy median {np.median(pol):.3f}, MoveIt median {np.median(mov):.3f} '
                  f'(MoveIt\'s chosen arm; {len(both)} trials)')
    done = [r for r in executed if r['success'] and r.get('moveit_duration') is not None]
    if done:
        total = np.array([r['policy_decision_time'] + r['policy_duration'] for r in done])
        base = np.array([r['moveit_decision_time'] + r['moveit_duration'] for r in done])
        print(f'decision + motion, completed trials: policy median {np.median(total):.2f} s, '
              f'MoveIt median {np.median(base):.2f} s')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plans', required=True, help='evaluate_policy.py --save-trajectories output')
    p.add_argument('--data', default=str(Path.home() / 'panda_data'))
    p.add_argument('--out', default=str(Path.home() / 'exec_results.jsonl'))
    p.add_argument('--max-trials', type=int)
    p.add_argument('--fps', type=int, default=5)
    p.add_argument('--speed', type=float, default=0.5)
    p.add_argument('--accel', type=float, default=0.5)
    p.add_argument('--exact-check', action='store_true',
                   help='execute the first candidate with no MoveIt contact (needs candidates in --plans)')
    p.add_argument('--retime', action='store_true', help='skip the frames where the arm barely moves')
    args = p.parse_args()

    data = Path(args.data).expanduser()
    plans = [json.loads(line) for line in Path(args.plans).expanduser().read_text().splitlines() if line]
    plans = plans[:args.max_trials] if args.max_trials else plans
    out_path = Path(args.out).expanduser()
    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line] if out_path.exists() else []
    done = {(r['seed'], r['task_index']) for r in rows}

    rclpy.init()
    applier = ScenarioApplier(args.speed, args.accel)
    logs = {name: JointLog(node) for name, node in applier.arms.items()}
    vmax = velocity_limits(args.speed)
    executor = SingleThreadedExecutor()
    for node in applier.arms.values():
        executor.add_node(node)
    spin = Thread(target=executor.spin, daemon=True)
    spin.start()
    checker = SceneChecker(applier)
    try:
        with out_path.open('a') as out:
            for n, plan in enumerate(plans, 1):
                key = (plan['seed'], plan['task_index'])
                if key in done:
                    continue
                trial = json.loads((data / f"trials/seed_{plan['seed']}_task_{plan['task_index']}.json")
                                   .read_text())
                moveit = trial['arms'].get(plan['moveit_arm'] or '', {})
                # MoveIt's decision: one plan per arm (the labels keep the best of
                # several repeats, so its cost and duration are the best case).
                plan_times = [a['single_plan_wall_time'] for a in trial['arms'].values()
                              if a.get('single_plan_wall_time') is not None]
                row = dict(seed=plan['seed'], task_index=plan['task_index'],
                           policy_arm=plan['policy_arm'], moveit_arm=plan['moveit_arm'],
                           policy_decision_time=plan['sample_time'] + plan['check_time'],
                           moveit_decision_time=sum(plan_times) if plan_times else None,
                           moveit_duration=moveit.get('best_trajectory_duration'),
                           moveit_cost=moveit.get('best_cost'))
                if plan['policy_arm']:
                    if 'trajectory' not in plan:
                        raise SystemExit('--plans has no trajectories; rerun evaluate_policy.py '
                                         'with --save-trajectories')
                    if args.exact_check and 'candidates' not in plan:
                        raise SystemExit('--exact-check needs candidates; rerun evaluate_policy.py '
                                         'with --save-trajectories')
                    scenario = json.loads((data / trial['scenario']).read_text())
                    try:
                        row.update(execute(applier, checker, logs, plan, trial, scenario, args, vmax))
                    except Exception as e:  # unattended: record it and go on
                        row.update(arm=plan['policy_arm'], success=False, failed_stage='error',
                                   error=f'{type(e).__name__}: {e}', moveit_contacts=[])
                    if row.get('exact_check_time') is not None:
                        row['policy_decision_time'] += row['exact_check_time']
                    print(f"seed {row['seed']} task {row['task_index']} {row['policy_arm']}"
                          f"{' (candidate ' + str(row['candidate']) + ')' if row.get('candidate') else ''}: "
                          f"{'ok' if row['success'] else 'failed at ' + str(row['failed_stage'])}"
                          f", contacts {row.get('moveit_contacts')}, tracking "
                          f"{row.get('tracking_error', 0):.3f} rad", flush=True)
                out.write(json.dumps(row) + '\n')
                out.flush()
                rows.append(row)
    finally:
        executor.shutdown()
        for node in applier.arms.values():
            node.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=2)
    summarize(rows)


if __name__ == '__main__':
    main()
