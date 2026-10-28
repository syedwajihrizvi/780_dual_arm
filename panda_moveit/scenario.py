#!/usr/bin/env python3
"""Seeded scenario randomizer for the dual-Panda table.

A scenario fixes everything that varies between trials: the two drop-off
trays, 0-4 obstacles (walls and blocks), all 12 cube poses and both arms'
start configurations. The same seed always gives the same scenario, and a
saved scenario file re-applies exactly, so both arms can be run on the
identical scene for paired trials.

    scenario.py --seed 7                   # generate, apply, save
    scenario.py --file scenario_7.json     # re-apply a saved scenario
    scenario.py --seed 7 --dry-run         # generate and print only

Applying moves both arms to 'ready' first so new obstacles cannot spawn
inside them, removes the previous obstacles and trays, teleports the cubes,
spawns the new obstacles and trays, syncs MoveIt's planning scene, then
moves the arms to start configurations MoveIt reports as collision-free.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path
from threading import Thread

import rclpy
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import (CollisionObject, PlanningScene,
                             PlanningSceneComponents)
from moveit_msgs.srv import GetPlanningScene, GetStateValidity

# gz.msgs types come from pick_place, which imports them in a safe order.
from pick_place import (ARMS, CUBE_SIZE, CUBES, Empty, Entity, EntityFactory,
                        PickPlace, Scene, gz_model_poses, gz_node, gz_request,
                        gz_set_pose, wait_done, yaw_quat)

TABLE_Z = 1.025
CUBE_Z = TABLE_Z + CUBE_SIZE / 2 + 0.0005
BASES = {'first': (0.2, 0.0), 'second': (1.5, 0.0)}
SCENE_PREFIXES = ('obstacle_', 'dropoff_')  # models a scenario replaces

# Sampling regions and clearances (metres). Footprints are oriented rectangles.
REGION_X = (0.40, 1.30)
REGION_Y = (-0.48, 0.48)           # table is +-0.6
CUBE_REACH = (0.30, 0.72)          # top-down grasp radius from a base
# Cubes both arms can reach make the interesting assignments (obstacles and
# path cost decide, not reach), so a share of the cubes is forced into that
# region. 0.76 m: Panda 1 managed one cube at 0.80 m and missed another.
SHARED_REACH = (0.30, 0.76)
SHARED_CUBES = 6
TRAY_REACH = (0.35, 0.78)          # trays must be reachable by both arms
TRAY_OUTER = 0.18
TRAY_GAP = 0.25                    # tray centre to tray centre
BASE_KEEPOUT = 0.25                # obstacle footprint to arm base axis
OBSTACLE_GAP = 0.12                # obstacle to obstacle / tray footprint
CUBE_OBSTACLE_CLEAR = 0.08         # open fingers reach 0.066 m from the cube
CUBE_TRAY_CLEAR = 0.06
CUBE_GAP = 0.12                    # cube centre to cube centre
OBSTACLE_COUNT = (3, 6)

# 'ready' from the SRDF; both arms use the same joint values.
READY = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
START_NOISE = [0.8, 0.4, 0.5, 0.5, 0.5, 0.5, 0.8]
JOINT_LIMITS = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
                (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525),
                (-2.8973, 2.8973)]


# ---------------------------------------------------------------
# Footprint geometry
# ---------------------------------------------------------------

def point_rect_distance(px, py, rect):
    """Distance from a point to an oriented rectangle (0 inside)."""
    cx, cy, sx, sy, yaw = rect
    c, s = math.cos(yaw), math.sin(yaw)
    lx = c * (px - cx) + s * (py - cy)
    ly = -s * (px - cx) + c * (py - cy)
    return math.hypot(max(abs(lx) - sx / 2, 0.0), max(abs(ly) - sy / 2, 0.0))


def boundary(rect, step=0.01):
    cx, cy, sx, sy, yaw = rect
    c, s = math.cos(yaw), math.sin(yaw)
    nx, ny = max(int(sx / step), 1), max(int(sy / step), 1)
    local = ([(-sx / 2 + sx * i / nx, sgn * sy / 2) for i in range(nx + 1) for sgn in (-1, 1)]
             + [(sgn * sx / 2, -sy / 2 + sy * j / ny) for j in range(ny + 1) for sgn in (-1, 1)])
    return [(cx + c * x - s * y, cy + s * x + c * y) for x, y in local]


def rect_distance(a, b):
    """Distance between two oriented rectangles, to ~1 cm (0 if overlapping).

    Sampling both boundaries also catches crossing walls with no corner inside.
    """
    return min(min(point_rect_distance(x, y, b) for x, y in boundary(a)),
               min(point_rect_distance(x, y, a) for x, y in boundary(b)))


def in_region(rect):
    return all(REGION_X[0] - 0.05 <= x <= REGION_X[1] + 0.05
               and REGION_Y[0] - 0.05 <= y <= REGION_Y[1] + 0.05
               for x, y in boundary(rect, step=1.0))


def footprint(obj):
    return (obj['xyz'][0], obj['xyz'][1], obj['size'][0], obj['size'][1], obj['yaw'])


# ---------------------------------------------------------------
# Generation (pure Python, deterministic in the seed)
# ---------------------------------------------------------------

def sample_trays(rng):
    trays = []
    while len(trays) < 2:
        x, y = rng.uniform(*REGION_X), rng.uniform(*REGION_Y)
        if not all(TRAY_REACH[0] <= math.dist((x, y), b) <= TRAY_REACH[1]
                   for b in BASES.values()):
            continue
        if any(math.dist((x, y), t['xyz'][:2]) < TRAY_GAP for t in trays):
            continue
        tray = dict(name=f"dropoff_{'ab'[len(trays)]}", xyz=[x, y, TABLE_Z],
                    size=[TRAY_OUTER, TRAY_OUTER, 0.03], yaw=rng.uniform(0, math.pi / 2))
        if in_region(footprint(tray)):
            trays.append(tray)
    return trays


def sample_obstacles(rng, trays, attempts=500):
    count = rng.randint(*OBSTACLE_COUNT)
    obstacles = []
    for _ in range(attempts):
        if len(obstacles) == count:
            break
        if rng.random() < 0.5:
            kind = 'wall'
            size = [rng.uniform(0.15, 0.35), rng.uniform(0.02, 0.04), rng.uniform(0.10, 0.30)]
        else:
            kind = 'block'
            size = [rng.uniform(0.06, 0.15), rng.uniform(0.06, 0.15), rng.uniform(0.06, 0.20)]
        x, y = rng.uniform(*REGION_X), rng.uniform(*REGION_Y)
        obj = dict(name=f'obstacle_{kind}_{len(obstacles)}', kind=kind, size=size,
                   xyz=[x, y, TABLE_Z + size[2] / 2], yaw=rng.uniform(0, math.pi))
        rect = footprint(obj)
        if not in_region(rect):
            continue
        if any(point_rect_distance(*b, rect) < BASE_KEEPOUT for b in BASES.values()):
            continue
        if any(rect_distance(rect, footprint(o)) < OBSTACLE_GAP for o in obstacles + trays):
            continue
        obstacles.append(obj)
    return obstacles if len(obstacles) == count else None


def sample_cubes(rng, trays, obstacles, attempts=3000):
    cubes = []
    names = sorted(CUBES)
    for _ in range(attempts):
        if len(cubes) == len(names):
            break
        x, y = rng.uniform(*REGION_X), rng.uniform(*REGION_Y)
        reach, need = (SHARED_REACH, all) if len(cubes) < SHARED_CUBES else (CUBE_REACH, any)
        if not need(reach[0] <= math.dist((x, y), b) <= reach[1] for b in BASES.values()):
            continue
        if any(point_rect_distance(x, y, footprint(o)) < CUBE_OBSTACLE_CLEAR for o in obstacles):
            continue
        if any(point_rect_distance(x, y, footprint(t)) < CUBE_TRAY_CLEAR for t in trays):
            continue
        if any(math.dist((x, y), c['xyz'][:2]) < CUBE_GAP for c in cubes):
            continue
        cubes.append(dict(name=names[len(cubes)], xyz=[x, y, CUBE_Z],
                          yaw=rng.uniform(-math.pi / 4, math.pi / 4)))
    return cubes if len(cubes) == len(names) else None


def generate(seed):
    """Trays, obstacles and cubes for a seed; arm starts are chosen on apply."""
    rng = random.Random(seed)
    for _ in range(100):
        trays = sample_trays(rng)
        obstacles = sample_obstacles(rng, trays)
        if obstacles is None:
            continue
        cubes = sample_cubes(rng, trays, obstacles)
        if cubes is not None:
            return dict(seed=seed, trays=trays, obstacles=obstacles, cubes=cubes)
    raise RuntimeError(f'could not generate a scenario for seed {seed}')


def sample_arm_start(rng):
    return [min(max(r + rng.uniform(-n, n), lo), hi)
            for r, n, (lo, hi) in zip(READY, START_NOISE, JOINT_LIMITS)]


# ---------------------------------------------------------------
# Applying a scenario to Gazebo and MoveIt
# ---------------------------------------------------------------

def box_sdf(name, boxes, rgb):
    """Static model of boxes [(size, xyz)] in the model frame."""
    parts = ''
    for i, (size, xyz) in enumerate(boxes):
        geom = f'<geometry><box><size>{" ".join(map(str, size))}</size></box></geometry>'
        pose = f'<pose>{" ".join(map(str, xyz))} 0 0 0</pose>'
        parts += (f'<collision name="c{i}">{pose}{geom}</collision>'
                  f'<visual name="v{i}">{pose}{geom}<material>'
                  f'<ambient>{rgb} 1</ambient><diffuse>{rgb} 1</diffuse></material></visual>')
    return (f'<sdf version="1.9"><model name="{name}"><static>true</static>'
            f'<link name="link">{parts}</link></model></sdf>')


def tray_boxes():
    """Tray walls in the tray frame (origin on the table top), as in the world SDF."""
    h, t, inner = 0.03, 0.01, TRAY_OUTER - 0.02
    off = (TRAY_OUTER - t) / 2
    return [([TRAY_OUTER, t, h], [0, -off, h / 2]), ([TRAY_OUTER, t, h], [0, off, h / 2]),
            ([t, inner, h], [-off, 0, h / 2]), ([t, inner, h], [off, 0, h / 2])]


def model_boxes(obj):
    """(sdf, boxes in model frame) for an obstacle or tray."""
    if obj['name'].startswith('dropoff_'):
        boxes = tray_boxes()
        return box_sdf(obj['name'], boxes, '0.8 0.8 0.8'), boxes
    boxes = [(obj['size'], [0, 0, 0])]
    return box_sdf(obj['name'], boxes, '0.45 0.5 0.55'), boxes


class ScenarioApplier:
    def __init__(self, speed, accel):
        self.arms = {name: PickPlace(arm, speed, accel) for name, arm in ARMS.items()}
        node = self.arms['first']
        self.get_scene = node.create_client(GetPlanningScene, '/get_planning_scene')
        self.validity = node.create_client(GetStateValidity, '/check_state_validity')

    @staticmethod
    def call(client, req, timeout=10):
        # A request sent before the service is discovered is silently lost.
        if not client.wait_for_service(timeout_sec=15.0):
            raise RuntimeError(f'{client.srv_name} unavailable')
        return wait_done(client.call_async(req), timeout)

    # -- Gazebo --------------------------------------------------

    def gz_models(self):
        return [m.name for m in gz_request('scene/info', Empty(), Empty, Scene).model]

    def gz_remove(self, name):
        req = Entity()
        req.name = name
        req.type = Entity.MODEL
        gz_request('remove', req, Entity)

    def gz_spawn(self, obj):
        req = EntityFactory()
        req.sdf = model_boxes(obj)[0]
        req.name = obj['name']
        req.pose.position.x, req.pose.position.y, req.pose.position.z = obj['xyz']
        _, _, req.pose.orientation.z, req.pose.orientation.w = yaw_quat(obj['yaw'])
        gz_request('create', req, EntityFactory)

    def release_cubes(self):
        """Detach every cube from both hands, in Gazebo and MoveIt."""
        pubs = [gz_node().advertise(f'/{arm.gz_model}/{cube}/detach', Empty)
                for arm in ARMS.values() for cube in CUBES]
        time.sleep(0.5)  # discovery; a message sent before it is dropped
        for pub in pubs:
            if pub.has_connections():
                pub.publish(Empty())
        for node in self.arms.values():
            node.open_gripper()

        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        attached = self.call(self.get_scene, req).scene.robot_state.attached_collision_objects
        if attached:
            scene = PlanningScene()
            for aco in attached:
                aco.object.operation = CollisionObject.REMOVE
                scene.robot_state.attached_collision_objects.append(aco)
            self.arms['first'].apply_scene(scene)

    # -- MoveIt --------------------------------------------------

    def moveit_world_names(self):
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
        scene = self.call(self.get_scene, req).scene
        return [o.id for o in scene.world.collision_objects]

    def moveit_sync(self, scenario):
        scene = PlanningScene()
        for name in self.moveit_world_names():
            if name.startswith(SCENE_PREFIXES):
                obj = CollisionObject()
                obj.id = name
                obj.operation = CollisionObject.REMOVE
                scene.world.collision_objects.append(obj)
        for item in scenario['trays'] + scenario['obstacles']:
            obj = CollisionObject()
            obj.id = item['name']
            obj.header.frame_id = 'world'
            obj.pose.position.x, obj.pose.position.y, obj.pose.position.z = item['xyz']
            (obj.pose.orientation.x, obj.pose.orientation.y,
             obj.pose.orientation.z, obj.pose.orientation.w) = yaw_quat(item['yaw'])
            for size, xyz in model_boxes(item)[1]:
                box = SolidPrimitive()
                box.type = SolidPrimitive.BOX
                box.dimensions = [float(v) for v in size]
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = map(float, xyz)
                pose.orientation.w = 1.0
                obj.primitives.append(box)
                obj.primitive_poses.append(pose)
            obj.operation = CollisionObject.ADD
            scene.world.collision_objects.append(obj)
        self.arms['first'].apply_scene(scene)
        for cube in scenario['cubes']:
            self.arms['first'].scene_set_cube(cube['name'], cube['xyz'], cube['yaw'])

    def state_valid(self, starts):
        req = GetStateValidity.Request()
        req.group_name = 'both_panda_arms'
        req.robot_state.is_diff = True
        names, positions = [], []
        for arm, joints in starts.items():
            names += [f'{arm}_panda_joint{i}' for i in range(1, 8)]
            positions += [float(v) for v in joints]
        req.robot_state.joint_state.name = names
        req.robot_state.joint_state.position = positions
        return self.call(self.validity, req).valid

    def choose_arm_starts(self, seed, attempts=200):
        rng = random.Random(f'{seed}-arm-starts')
        for _ in range(attempts):
            starts = {arm: sample_arm_start(rng) for arm in ARMS}
            if self.state_valid(starts):
                return starts
        raise RuntimeError('no collision-free arm start configuration found')

    # -- Whole scenario -------------------------------------------

    def move_arms(self, starts):
        for arm, joints in starts.items():
            record = self.arms[arm].move_to_joints(joints)
            if not record['success']:
                # A stuck or out-of-limits arm makes every later plan fail;
                # drive it to READY directly, then plan again.
                print(f'{arm} arm: move failed (MoveIt code {record["error_code"]}), recovering')
                if not self.arms[arm].drive_to_joints(READY):
                    # Seen once: Gazebo locks a joint at its hard limit. Only a
                    # sim restart frees it. SystemExit gets past per-seed retries.
                    raise SystemExit(f'{arm} arm is stuck in Gazebo (joint at a hard limit); '
                                     'restart the sim, then rerun to resume')
                record = self.arms[arm].move_to_joints(joints)
            if not record['success']:
                raise RuntimeError(f'{arm} arm could not reach {joints} '
                                   f'(MoveIt code {record["error_code"]})')

    def apply(self, scenario):
        self.release_cubes()
        self.move_arms({arm: READY for arm in ARMS})

        for name in self.gz_models():
            if name.startswith(SCENE_PREFIXES):
                self.gz_remove(name)
        # Cubes before spawning, so none is teleported into a new obstacle.
        for cube in scenario['cubes']:
            gz_set_pose(cube['name'], cube['xyz'], cube['yaw'])
        for item in scenario['trays'] + scenario['obstacles']:
            self.gz_spawn(item)
        self.moveit_sync(scenario)

        if 'arm_starts' not in scenario:
            scenario['arm_starts'] = self.choose_arm_starts(scenario['seed'])
        self.move_arms(scenario['arm_starts'])

        time.sleep(0.5)  # let the cubes settle
        poses = gz_model_poses()
        drift = max(math.dist(poses[c['name']][0], c['xyz']) for c in scenario['cubes'])
        scenario['max_cube_drift'] = drift
        if drift > 0.01:
            raise RuntimeError(f'a cube moved {drift:.3f} m after applying; scene disturbed')
        return scenario


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--seed', type=int, help='generate a new scenario from this seed')
    src.add_argument('--file', help='re-apply a saved scenario JSON')
    p.add_argument('--dry-run', action='store_true', help='print the generated scenario only')
    p.add_argument('--out', help='default: /tmp/scenarios/scenario_<seed>.json')
    p.add_argument('--speed', type=float, default=0.5)
    p.add_argument('--accel', type=float, default=0.5)
    args = p.parse_args()

    scenario = json.loads(Path(args.file).read_text()) if args.file else generate(args.seed)
    if args.dry_run:
        print(json.dumps(scenario, indent=2))
        return

    rclpy.init()
    applier = ScenarioApplier(args.speed, args.accel)
    executor = SingleThreadedExecutor()
    for node in applier.arms.values():
        executor.add_node(node)
    spin = Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        start = time.monotonic()
        applier.apply(scenario)
        out = Path(args.out or args.file or f'/tmp/scenarios/scenario_{scenario["seed"]}.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(scenario, indent=2) + '\n')
        print(f"Applied scenario {scenario['seed']}: {len(scenario['obstacles'])} obstacles, "
              f"trays at {[t['xyz'][:2] for t in scenario['trays']]} "
              f"in {time.monotonic() - start:.1f}s; saved {out}")
    finally:
        executor.shutdown()
        for node in applier.arms.values():
            node.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=2)


if __name__ == '__main__':
    main()
