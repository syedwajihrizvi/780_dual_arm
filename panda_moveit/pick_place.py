#!/usr/bin/env python3
"""Panda pick-and-place pipeline, for either arm (--arm first|second).

pregrasp -> open -> descend -> close -> attach -> lift -> transport
-> lower -> detach -> open -> retreat

Plan only by default; stages are chained so each one plans from the
previous stage's end state (the cube is attached in MoveIt temporarily
and the planning scene is restored afterwards). --execute moves the
simulated arm and holds the cube with a Gazebo DetachableJoint that is
attached only after the closed fingers confirm the cube is between them.
"""
import argparse
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.msg import (AttachedCollisionObject, BoundingVolume,
                             CollisionObject, Constraints, JointConstraint,
                             MoveItErrorCodes,
                             MotionPlanRequest, OrientationConstraint,
                             PlanningOptions, PlanningScene,
                             PositionConstraint, RobotState)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPositionIK
from std_msgs.msg import Float64MultiArray
from gz.transport13 import Node as GzNode
# Keep all gz.msgs imports here, Pose_V first: importing entity_factory_pb2
# before it makes received Pose_V messages undecodable ("No message class
# registered for 'gz.msgs.Pose'"). scenario.py imports them from here.
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.entity_plugin_v_pb2 import EntityPlugin_V
from gz.msgs10.pose_pb2 import Pose as GzPose
from gz.msgs10.scene_pb2 import Scene
from gz.msgs10.stringmsg_pb2 import StringMsg

GZ_WORLD = 'panda_world'
GZ_HAND_LINK = 'panda_link7'  # panda_hand is merged into link7 in the SDF

# Grasp point (centre of finger pads) along link8 +Z: Franka flange-to-TCP.
# Fingertips are at 0.1122 m (0.0584 joint offset + 0.0538 finger mesh).
TCP_OFFSET = 0.1034


@dataclass(frozen=True)
class Arm:
    name: str           # MoveIt prefix: '<name>_panda_...'
    gz_model: str       # Gazebo model name
    ns: str             # ROS namespace of the arm's Gazebo controllers
    link8_down: tuple   # xyzw grasp orientation of link8 in world
    default_cube: str
    default_dropoff: str

    @property
    def group(self):
        return f'{self.name}_panda_arm'

    @property
    def link8(self):
        return f'{self.name}_panda_link8'

    @property
    def hand(self):
        return f'{self.name}_panda_hand'

    @property
    def finger_joint(self):
        return f'{self.name}_panda_finger_joint1'  # finger_joint2 mimics it in MoveIt

    @property
    def touch_links(self):
        return [self.hand, f'{self.name}_panda_leftfinger', f'{self.name}_panda_rightfinger']


# link8 pointing down, RPY (pi, 0, -pi/4) for Panda 1: the hand frame is then
# aligned with world X/Y, so the fingers close along world Y onto the cube
# faces. Panda 2's base is yawed by pi, so its grasp is the same rotated by pi.
ARMS = {
    'first': Arm('first', 'panda', '', (0.9238795, -0.3826834, 0.0, 0.0),
                 'cube_red', 'dropoff_a'),
    'second': Arm('second', 'panda_second', '/second', (0.3826834, 0.9238795, 0.0, 0.0),
                  'cube_green', 'dropoff_b'),
}

# Every planning request uses these; trials store them so the MoveIt
# baseline (its decision time and costs) is reported with its settings.
# OMPL planner and pipeline are set in the launch file.
PLANNER_SETTINGS = dict(
    pipeline='ompl',
    planner='RRTConnect',
    allowed_planning_time=8.0,
    num_planning_attempts=10,
    goal_position_tolerance=0.005,       # m, radius of the link8 goal sphere
    goal_orientation_tolerance=0.05,     # rad, per axis
    cartesian_max_step=0.005,            # m
    cartesian_min_fraction=0.99,
    ik_precheck_timeout=0.5,             # s, /compute_ik before OMPL
    ik_solver='KDL',
)

CUBE_SIZE = 0.04
GRIPPER_OPEN = 0.04
GRIPPER_CLOSE = 0.015  # below half the cube width, so the fingers squeeze
# The fingers stall with their gap (finger1 + finger2) near the cube width.
# Check the gap, not each finger: the cube is rarely centred, e.g. 0.0237/0.0165.
GRASP_MIN_GAP = CUBE_SIZE - 0.004

WORLD_FILE = Path(get_package_share_directory('panda_moveit')) / 'worlds' / 'panda_world.sdf'


def world_positions(prefix):
    """Initial xyz of every world model whose name starts with prefix."""
    world = ET.parse(WORLD_FILE).getroot().find('world')
    return {m.get('name'): tuple(float(v) for v in m.find('pose').text.split()[:3])
            for m in world.findall('model') if m.get('name', '').startswith(prefix)}


# Initial cube centres (all yaw 0) and drop-off tray centres on the table top.
CUBES = world_positions('cube_')
DROPOFFS = world_positions('dropoff_')


def rotate(q, v):
    """Rotate vector v by unit quaternion q (xyzw)."""
    x, y, z, w = q
    # v' = v + 2w(u x v) + 2u x (u x v), u = (x, y, z)
    cx = y * v[2] - z * v[1]
    cy = z * v[0] - x * v[2]
    cz = x * v[1] - y * v[0]
    return (v[0] + 2 * (w * cx + y * cz - z * cy),
            v[1] + 2 * (w * cy + z * cx - x * cz),
            v[2] + 2 * (w * cz + x * cy - y * cx))


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def grasp_yaws(cube_yaw):
    """Hand yaws (from the nominal grasp) that close the fingers square on a cube.

    The cube looks the same every 90 deg: first the equivalent yaw closest to
    zero, then the one turned 90 deg, which grips the other pair of faces.
    The hand body is 0.21 m wide across the fingers but 0.06 m along them, so
    the turned grasp often fits beside an obstacle where the first does not.
    """
    yaw = (cube_yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
    return [yaw, yaw - math.pi / 2 if yaw > 0 else yaw + math.pi / 2]


def grasp_orientation(arm, hand_yaw):
    return quat_mul(yaw_quat(hand_yaw), arm.link8_down)


def link8_pose(tcp_xyz, q):
    """T_world_link8 = T_world_tcp * inverse(T_link8_tcp)."""
    offset = rotate(q, (0.0, 0.0, TCP_OFFSET))
    pose = Pose()
    pose.position.x = tcp_xyz[0] - offset[0]
    pose.position.y = tcp_xyz[1] - offset[1]
    pose.position.z = tcp_xyz[2] - offset[2]
    (pose.orientation.x, pose.orientation.y,
     pose.orientation.z, pose.orientation.w) = q
    return pose


def above(xyz, dz):
    return (xyz[0], xyz[1], xyz[2] + dz)


def metrics(result_code, trajectory, planning_time=0.0):
    points = trajectory.joint_trajectory.points
    path_length = 0.0
    for a, b in zip(points, points[1:]):
        path_length += sum((x - y) ** 2 for x, y in zip(a.positions, b.positions)) ** 0.5
    duration = 0.0
    if points:
        stamp = points[-1].time_from_start
        duration = stamp.sec + stamp.nanosec * 1e-9
    return dict(success=result_code == MoveItErrorCodes.SUCCESS,
                error_code=result_code,
                planning_time=planning_time,
                trajectory_duration=duration,
                joint_path_length=path_length,
                trajectory_points=len(points))


def end_state(trajectory, finger_joint, finger):
    """RobotState at the end of a trajectory, as a diff on the current state."""
    state = RobotState()
    state.is_diff = True
    jt = trajectory.joint_trajectory
    state.joint_state.name = list(jt.joint_names) + [finger_joint]
    state.joint_state.position = list(jt.points[-1].positions) + [finger]
    return state


def cube_box():
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [CUBE_SIZE] * 3
    return box


# ---------------------------------------------------------------
# Gazebo helpers (gz-transport; the gz CLI is slow and occasionally hangs)
# ---------------------------------------------------------------

_gz_node = None


def gz_node():
    global _gz_node
    if _gz_node is None:
        _gz_node = GzNode()
    return _gz_node


def gz_request(service, req, req_type, rep_type=Boolean, timeout_ms=5000):
    ok, rep = gz_node().request(f'/world/{GZ_WORLD}/{service}', req, req_type,
                                rep_type, timeout_ms)
    if not ok:
        raise RuntimeError(f'Gazebo {service} request failed')
    return rep


def gz_model_id(name):
    for model in gz_request('scene/info', Empty(), Empty, Scene).model:
        if model.name == name:
            return model.id
    raise KeyError(f'no Gazebo model {name}')


def gz_model_poses(timeout=3.0):
    """Live {model name: (xyz, yaw)} from one pose/info message."""
    topic = f'/world/{GZ_WORLD}/pose/info'
    received = Event()
    poses = {}

    def on_poses(msg):
        # Read the message inside the callback: it cannot be decoded after
        # the subscription is gone.
        if received.is_set():
            return
        for p in msg.pose:
            q = p.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            # Models come before their links, and link names repeat across arms.
            poses.setdefault(p.name, ((p.position.x, p.position.y, p.position.z), yaw))
        received.set()

    node = GzNode()
    node.subscribe(Pose_V, topic, on_poses)
    try:
        if not received.wait(timeout):
            raise TimeoutError(f'no message on {topic}')
    finally:
        node.unsubscribe(topic)
    return poses


def gz_model_pose(name):
    """Live (xyz, yaw) of a Gazebo model."""
    return gz_model_poses()[name]


def gz_set_pose(name, xyz, yaw=0.0):
    req = GzPose()
    req.name = name
    req.position.x, req.position.y, req.position.z = xyz
    _, _, req.orientation.z, req.orientation.w = yaw_quat(yaw)
    gz_request('set_pose', req, GzPose)


class GzGrip:
    """DetachableJoint between the Panda hand and one cube.

    The system is added to the robot at runtime on first attach (it attaches
    as soon as it loads) and toggled through its topics afterwards. Uses
    gz-transport directly: the gz CLI costs ~1 s per attach/detach.
    """

    def __init__(self, arm, cube):
        self.gz_model = arm.gz_model
        self.cube = cube
        prefix = f'/{arm.gz_model}/{cube}'
        self.attach_topic = f'{prefix}/attach'
        self.detach_topic = f'{prefix}/detach'
        self.state_topic = f'{prefix}/state'
        self.node = GzNode()
        self.attach_pub = self.node.advertise(self.attach_topic, Empty)
        self.detach_pub = self.node.advertise(self.detach_topic, Empty)
        self.state = None  # last 'attached'/'detached' report
        self.state_changed = Event()
        self.node.subscribe(StringMsg, self.state_topic, self._on_state)

    def _on_state(self, msg):
        self.state = msg.data
        self.state_changed.set()

    def loaded(self, wait=0.5):
        """The system subscribes to its detach topic once loaded."""
        deadline = time.monotonic() + wait
        while not self.detach_pub.has_connections():
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True

    def _load(self):
        req = EntityPlugin_V()
        req.entity.id = gz_model_id(self.gz_model)  # lookup by name is silently ignored
        plugin = req.plugins.add()
        plugin.name = 'gz::sim::systems::DetachableJoint'
        plugin.filename = 'gz-sim-detachable-joint-system'
        plugin.innerxml = (f'<parent_link>{GZ_HAND_LINK}</parent_link>'
                           f'<child_model>{self.cube}</child_model>'
                           f'<child_link>cube_link</child_link>'
                           f'<attach_topic>{self.attach_topic}</attach_topic>'
                           f'<detach_topic>{self.detach_topic}</detach_topic>'
                           f'<output_topic>{self.state_topic}</output_topic>')
        gz_request('entity/system/add', req, EntityPlugin_V)

    def _toggle(self, want, trigger, timeout):
        """Run trigger and wait for the state topic to report `want`."""
        start = time.monotonic()
        self.state_changed.clear()
        trigger()
        while self.state != want and time.monotonic() - start < timeout:
            self.state_changed.wait(0.02)
            self.state_changed.clear()
        return dict(success=self.state == want, gz_state=self.state,
                    duration=time.monotonic() - start)

    def attach(self, timeout=3.0):
        # run() creates the GzGrip seconds before the grasp, so discovery of
        # an already loaded system is done by now; no need to wait for it.
        if self.loaded(wait=0.1):
            return self._toggle('attached', lambda: self.attach_pub.publish(Empty()), timeout)
        record = self._toggle('attached', self._load, timeout)
        if not record['success'] and self.loaded():
            # Its first report can go out before our subscriber connects.
            record.update(success=True, gz_state='attached on load')
        return record

    def detach(self, timeout=3.0):
        return self._toggle('detached', lambda: self.detach_pub.publish(Empty()), timeout)


class PickPlace(Node):
    def __init__(self, arm, speed, accel):
        super().__init__(f'panda_pick_place_{arm.name}')
        self.arm = arm
        # Fractions of the joint velocity / acceleration limits.
        self.speed = speed
        self.accel = accel
        self.move_client = ActionClient(self, MoveGroup, '/move_action')
        self.exec_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.controller = ActionClient(
            self, FollowJointTrajectory, f'{arm.ns}/panda_arm_controller/follow_joint_trajectory')
        self.cartesian = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.scene = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self.ik = self.create_client(GetPositionIK, '/compute_ik')
        self.gripper = self.create_publisher(
            Float64MultiArray, f'{arm.ns}/panda_gripper_controller/commands', 10)
        self.fingers = None
        self.create_subscription(JointState, f'{arm.ns}/joint_states', self.on_joints, 10)

    def on_joints(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if 'panda_finger_joint1' in pos and 'panda_finger_joint2' in pos:
            self.fingers = (pos['panda_finger_joint1'], pos['panda_finger_joint2'])

    # -- motion -------------------------------------------------

    def move_to_pose(self, pose, start_state=None, execute=False):
        """OMPL joint-space plan for link8 to reach pose."""
        target = PoseStamped()
        target.header.frame_id = 'world'
        target.pose = pose

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [PLANNER_SETTINGS['goal_position_tolerance']]
        volume = BoundingVolume()
        volume.primitives.append(sphere)
        volume.primitive_poses.append(target.pose)

        pc = PositionConstraint()
        pc.header = target.header
        pc.link_name = self.arm.link8
        pc.constraint_region = volume
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header = target.header
        oc.link_name = self.arm.link8
        oc.orientation = target.pose.orientation
        tolerance = PLANNER_SETTINGS['goal_orientation_tolerance']
        oc.absolute_x_axis_tolerance = tolerance
        oc.absolute_y_axis_tolerance = tolerance
        oc.absolute_z_axis_tolerance = tolerance  # yaw sets finger closing direction
        oc.weight = 1.0

        constraint = Constraints()
        constraint.position_constraints = [pc]
        constraint.orientation_constraints = [oc]

        req = MotionPlanRequest()
        req.group_name = self.arm.group
        req.num_planning_attempts = PLANNER_SETTINGS['num_planning_attempts']
        req.allowed_planning_time = PLANNER_SETTINGS['allowed_planning_time']
        req.max_velocity_scaling_factor = self.speed
        req.max_acceleration_scaling_factor = self.accel
        req.goal_constraints = [constraint]
        if start_state is not None:
            req.start_state = start_state
        else:
            # Start from the current measured robot state.
            req.start_state.is_diff = True

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = not execute
        goal.planning_options.look_around = False
        goal.planning_options.replan = False

        if not self.move_client.wait_for_server(timeout_sec=15.0):
            raise RuntimeError('/move_action unavailable; check move_group')
        handle = wait_done(self.move_client.send_goal_async(goal), 20)
        if not handle.accepted:
            raise RuntimeError('MoveIt rejected the goal')
        result = wait_done(handle.get_result_async(), 90).result
        return (metrics(result.error_code.val, result.planned_trajectory,
                        result.planning_time),
                result.planned_trajectory)

    def reachable(self, pose, finger=GRIPPER_OPEN,
                  timeout=PLANNER_SETTINGS['ik_precheck_timeout']):
        """Collision-free IK exists for link8 at pose, with the gripper at finger.

        Milliseconds, versus OMPL spending its whole planning time when the
        goal has no valid IK solution.
        """
        req = GetPositionIK.Request()
        ik = req.ik_request
        ik.group_name = self.arm.group
        ik.ik_link_name = self.arm.link8
        ik.pose_stamped.header.frame_id = 'world'
        ik.pose_stamped.pose = pose
        ik.avoid_collisions = True
        ik.timeout.nanosec = int(timeout * 1e9)
        ik.robot_state.is_diff = True
        ik.robot_state.joint_state.name = [self.arm.finger_joint]
        ik.robot_state.joint_state.position = [finger]
        if not self.ik.wait_for_service(timeout_sec=15.0):
            raise RuntimeError('/compute_ik unavailable')
        res = wait_done(self.ik.call_async(req), timeout + 10)
        return res.error_code.val == MoveItErrorCodes.SUCCESS

    def drive_to_joints(self, positions, duration=5.0):
        """Straight joint-space move sent to the controller, bypassing MoveIt.
        Recovery only: no collision checking, but it works from a state MoveIt
        rejects (e.g. Gazebo pushed a joint past its limit)."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [f'panda_joint{i}' for i in range(1, 8)]
        point = JointTrajectoryPoint(positions=[float(v) for v in positions])
        point.time_from_start = Duration(sec=int(duration), nanosec=int(duration % 1 * 1e9))
        goal.trajectory.points = [point]
        if not self.controller.wait_for_server(timeout_sec=15.0):
            raise RuntimeError(f'{self.arm.name} arm controller unavailable')
        handle = wait_done(self.controller.send_goal_async(goal), 20)
        if not handle.accepted:
            raise RuntimeError('controller rejected the recovery goal')
        return wait_done(handle.get_result_async(), duration + 20).result.error_code == 0

    def move_to_joints(self, positions, execute=True):
        """OMPL plan to a 7-joint arm configuration."""
        constraint = Constraints()
        for i, value in enumerate(positions, start=1):
            jc = JointConstraint()
            jc.joint_name = f'{self.arm.name}_panda_joint{i}'
            jc.position = float(value)
            jc.tolerance_above = jc.tolerance_below = 0.001
            jc.weight = 1.0
            constraint.joint_constraints.append(jc)

        req = MotionPlanRequest()
        req.group_name = self.arm.group
        req.num_planning_attempts = PLANNER_SETTINGS['num_planning_attempts']
        req.allowed_planning_time = PLANNER_SETTINGS['allowed_planning_time']
        req.max_velocity_scaling_factor = self.speed
        req.max_acceleration_scaling_factor = self.accel
        req.goal_constraints = [constraint]
        req.start_state.is_diff = True

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = not execute

        if not self.move_client.wait_for_server(timeout_sec=15.0):
            raise RuntimeError('/move_action unavailable; check move_group')
        handle = wait_done(self.move_client.send_goal_async(goal), 20)
        if not handle.accepted:
            raise RuntimeError('MoveIt rejected the goal')
        result = wait_done(handle.get_result_async(), 90).result
        return metrics(result.error_code.val, result.planned_trajectory,
                       result.planning_time)

    def cartesian_to(self, pose, start_state=None, execute=False,
                     min_fraction=PLANNER_SETTINGS['cartesian_min_fraction']):
        """Straight-line link8 motion to pose, collision checked."""
        req = GetCartesianPath.Request()
        req.header.frame_id = 'world'
        req.group_name = self.arm.group
        req.link_name = self.arm.link8
        req.waypoints = [pose]
        req.max_step = PLANNER_SETTINGS['cartesian_max_step']
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = self.speed
        req.max_acceleration_scaling_factor = self.accel
        if start_state is not None:
            req.start_state = start_state
        else:
            req.start_state.is_diff = True

        if not self.cartesian.wait_for_service(timeout_sec=15.0):
            raise RuntimeError('/compute_cartesian_path unavailable')
        start = time.monotonic()
        res = wait_done(self.cartesian.call_async(req), 30)
        planning_time = time.monotonic() - start  # the service reports none
        code = res.error_code.val
        if code == MoveItErrorCodes.SUCCESS and res.fraction < min_fraction:
            code = MoveItErrorCodes.PLANNING_FAILED
        record = metrics(code, res.solution, planning_time)
        record['fraction'] = res.fraction

        if execute and record['success']:
            if not self.exec_client.wait_for_server(timeout_sec=15.0):
                raise RuntimeError('/execute_trajectory unavailable')
            goal = ExecuteTrajectory.Goal()
            goal.trajectory = res.solution
            handle = wait_done(self.exec_client.send_goal_async(goal), 20)
            if not handle.accepted:
                raise RuntimeError('MoveIt rejected the trajectory')
            exec_code = wait_done(handle.get_result_async(), 90).result.error_code.val
            record['success'] = exec_code == MoveItErrorCodes.SUCCESS
            record['error_code'] = exec_code
        return record, res.solution

    # -- gripper ------------------------------------------------

    def command_gripper(self, width, timeout=5.0):
        # A message published before the controller's subscription is matched
        # is silently dropped, which happens on the first command after startup.
        start = time.monotonic()
        while self.gripper.get_subscription_count() == 0:
            if time.monotonic() - start > timeout:
                raise RuntimeError(f'no subscriber on {self.gripper.topic_name}')
            time.sleep(0.05)
        msg = Float64MultiArray()
        msg.data = [width, width]
        self.gripper.publish(msg)

    def open_gripper(self, width=GRIPPER_OPEN, timeout=10.0, tol=0.002):
        """Command both fingers and wait until /joint_states reaches them."""
        self.command_gripper(width)
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self.fingers and all(abs(f - width) < tol for f in self.fingers):
                return dict(success=True, fingers=list(self.fingers),
                            duration=time.monotonic() - start)
            time.sleep(0.05)
        return dict(success=False, fingers=list(self.fingers or []),
                    duration=timeout, reason='fingers did not reach target')

    def close_gripper(self, timeout=3.0, settle=0.15):
        """Squeeze, wait for the fingers to stall, and check a cube stopped them."""
        open_gap = sum(self.fingers)
        self.command_gripper(GRIPPER_CLOSE)
        start = time.monotonic()
        last, still = self.fingers, 0.0
        while time.monotonic() - start < timeout and still < settle:
            time.sleep(0.02)
            moved = max(abs(a - b) for a, b in zip(self.fingers, last))
            last = self.fingers
            # Count stillness only once the fingers have started closing,
            # otherwise the pause before they react looks like a stall.
            closing = sum(last) < open_gap - 0.005
            still = still + 0.02 if closing and moved < 5e-5 else 0.0
        gap = sum(self.fingers)
        held = gap > GRASP_MIN_GAP
        record = dict(success=held, fingers=list(self.fingers), gap=gap,
                      duration=time.monotonic() - start)
        if not held:
            record['reason'] = 'fingers closed past the cube width; nothing grasped'
        return record

    # -- planning scene -----------------------------------------

    def apply_scene(self, scene):
        scene.is_diff = True
        scene.robot_state.is_diff = True
        if not self.scene.wait_for_service(timeout_sec=15.0):
            raise RuntimeError('/apply_planning_scene unavailable')
        req = ApplyPlanningScene.Request()
        req.scene = scene
        if not wait_done(self.scene.call_async(req), 10).success:
            raise RuntimeError('apply_planning_scene failed')

    def scene_attach(self, cube):
        """Move the cube from the world onto the hand, at the grasp point."""
        aco = AttachedCollisionObject()
        aco.link_name = self.arm.hand
        aco.touch_links = self.arm.touch_links
        aco.object.id = cube
        aco.object.header.frame_id = self.arm.hand
        aco.object.primitives = [cube_box()]
        grasp = Pose()
        grasp.position.z = TCP_OFFSET
        grasp.orientation.w = 1.0  # hand frame is cube-aligned at grasp
        aco.object.primitive_poses = [grasp]
        # ADD also removes the same-named world object.
        aco.object.operation = CollisionObject.ADD
        scene = PlanningScene()
        scene.robot_state.attached_collision_objects = [aco]
        self.apply_scene(scene)

    def scene_detach(self, cube, world_xyz, yaw=0.0):
        """Drop the cube off the hand and put it in the world at world_xyz."""
        aco = AttachedCollisionObject()
        aco.link_name = self.arm.hand
        aco.object.id = cube
        aco.object.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.robot_state.attached_collision_objects = [aco]
        self.apply_scene(scene)
        self.scene_set_cube(cube, world_xyz, yaw)

    def scene_set_cube(self, cube, world_xyz, yaw=0.0):
        obj = CollisionObject()
        obj.id = cube
        obj.header.frame_id = 'world'
        obj.primitives = [cube_box()]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = world_xyz
        (pose.orientation.x, pose.orientation.y,
         pose.orientation.z, pose.orientation.w) = yaw_quat(yaw)
        obj.primitive_poses = [pose]
        obj.operation = CollisionObject.ADD  # replaces an existing object
        scene = PlanningScene()
        scene.world.collision_objects = [obj]
        self.apply_scene(scene)


def wait_done(future, timeout):
    start = time.monotonic()
    while not future.done():
        if time.monotonic() - start > timeout:
            raise TimeoutError('MoveIt request timed out')
        time.sleep(0.03)
    return future.result()


def reset_cube(node, cube):
    """Return a cube to its initial pose in Gazebo and MoveIt."""
    for arm in ARMS.values():
        grip = GzGrip(arm, cube)
        if grip.loaded():
            grip.detach(timeout=0.5)  # no report if it was already detached
    gz_set_pose(cube, CUBES[cube])
    node.scene_set_cube(cube, CUBES[cube])


def run(node, cube, cube_xyz, place_xyz, pregrasp_height, lift_height, execute,
        cube_yaw=0.0, quiet=False, keep_trajectories=False):
    """Run the stages in order, stopping at the first failure.

    Pick stages turn the hand to the cube's yaw; from transport on the hand
    returns to the nominal grasp, so the cube is placed square to the world.
    """
    arm = node.arm
    grip = GzGrip(arm, cube)
    stages = []
    state = None  # plan-only: chained start state; execute: current state

    def log(stage, record, tcp=None):
        entry = dict(stage=stage, **record)
        if tcp is not None:
            entry['tcp_xyz'] = list(tcp)
        if execute:
            entry['t_end'] = time.monotonic()  # lines stages up with a recording
        stages.append(entry)
        if not quiet:
            print(json.dumps(entry))
        return record['success']

    def move(stage, tcp, finger, cartesian=True, q=arm.link8_down):
        """finger: gripper width when the next motion stage starts."""
        nonlocal state
        plan = node.cartesian_to if cartesian else node.move_to_pose
        record, traj = plan(link8_pose(tcp, q), state, execute)
        if record['success'] and not execute:
            state = end_state(traj, arm.finger_joint, finger)
        if keep_trajectories:
            jt = traj.joint_trajectory
            record['trajectory'] = dict(
                joint_names=list(jt.joint_names),
                time_from_start=[p.time_from_start.sec + p.time_from_start.nanosec * 1e-9
                                 for p in jt.points],
                positions=[list(p.positions) for p in jt.points])
        return log(stage, record, tcp)

    attached_in_scene = False
    try:
        # Open first so a gripper left closed on a cube cannot drag it along.
        if execute and not log('open_gripper', node.open_gripper()):
            return stages
        # Pick the first grasp yaw whose pregrasp and descent both plan.
        pregrasp_tcp = above(cube_xyz, pregrasp_height)
        tried = []
        q_grasp = None
        select_start = time.monotonic()
        for hand_yaw in grasp_yaws(cube_yaw):
            q = grasp_orientation(arm, hand_yaw)
            if not (node.reachable(link8_pose(pregrasp_tcp, q))
                    and node.reachable(link8_pose(cube_xyz, q))):
                tried.append(dict(hand_yaw=hand_yaw, ik=False))
                continue
            record, traj = node.move_to_pose(link8_pose(pregrasp_tcp, q), state, False)
            fraction = 0.0
            if record['success']:
                check, _ = node.cartesian_to(link8_pose(cube_xyz, q),
                                             end_state(traj, arm.finger_joint, GRIPPER_OPEN))
                fraction = check['fraction']
                if check['success']:
                    q_grasp = q
            tried.append(dict(hand_yaw=hand_yaw, ik=True, pregrasp=record['success'],
                              descend_fraction=fraction))
            if q_grasp is not None:
                break
        # planning_time: IK checks plus the pregrasp and descent test plans.
        if not log('grasp_select', dict(success=q_grasp is not None, tried=tried,
                                        hand_yaw=tried[-1]['hand_yaw'],
                                        planning_time=time.monotonic() - select_start)):
            return stages

        if not move('pregrasp', pregrasp_tcp, GRIPPER_OPEN, cartesian=False, q=q_grasp):
            return stages
        if not move('descend', cube_xyz, CUBE_SIZE / 2, q=q_grasp):
            return stages

        if execute:
            if not log('close_gripper', node.close_gripper()):
                return stages
            if not log('gz_attach', grip.attach()):
                return stages
        node.scene_attach(cube)
        attached_in_scene = True

        if not move('lift', above(cube_xyz, lift_height), CUBE_SIZE / 2, q=q_grasp):
            return stages
        if not move('transport', above(place_xyz, lift_height), CUBE_SIZE / 2, cartesian=False):
            return stages
        place_tcp = above(place_xyz, 0.003)  # release just above the table
        if not move('lower', place_tcp, GRIPPER_OPEN):
            return stages

        if execute:
            if not log('gz_detach', grip.detach()):
                return stages
        node.scene_detach(cube, place_xyz)
        attached_in_scene = False
        if execute and not log('release', node.open_gripper()):
            return stages

        if not move('retreat', above(place_xyz, lift_height), GRIPPER_OPEN):
            return stages

        if execute:
            time.sleep(1.0)  # let the cube settle
            final, _ = gz_model_pose(cube)
            error = sum((a - b) ** 2 for a, b in zip(final, place_xyz)) ** 0.5
            log('verify_place', dict(success=error < 0.015, cube_final_xyz=list(final),
                                     place_error=error))
    finally:
        if not execute:
            # Plan-only must leave the planning scene as it found it.
            if attached_in_scene:
                node.scene_detach(cube, cube_xyz, cube_yaw)
            node.scene_set_cube(cube, cube_xyz, cube_yaw)
        elif attached_in_scene:
            print('WARNING: stopped while holding the cube; run with --reset to recover')
    return stages


# Task cost C = T_total + LAMBDA * L_total (seconds + seconds per radian).
LAMBDA = 1.0
COST_DEFINITION = f'T_total + {LAMBDA} * L_total'


def summarize(stages, execute):
    """Outcome and cost totals of one run() result; cost is None if it failed."""
    last = 'verify_place' if execute else 'retreat'
    success = bool(stages) and all(s['success'] for s in stages) and stages[-1]['stage'] == last
    motion = [s for s in stages if 'trajectory_duration' in s]
    duration = sum(s['trajectory_duration'] for s in motion)
    path_length = sum(s['joint_path_length'] for s in motion)
    return dict(success=success,
                failed_stage=None if success else stages[-1]['stage'],
                total_trajectory_duration=duration,
                total_joint_path_length=path_length,
                total_planning_time=sum(s.get('planning_time', 0.0) for s in stages),
                cost=duration + LAMBDA * path_length if success else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true', help='Move the real simulated arm; default plan-only')
    p.add_argument('--arm', default='first', choices=sorted(ARMS))
    p.add_argument('--cube', choices=sorted(CUBES), help="default: the arm's nearest cube")
    p.add_argument('--dropoff', choices=sorted(DROPOFFS),
                   help="tray to place the cube in; default: the arm's own tray")
    p.add_argument('--place', type=float, nargs=2, metavar=('X', 'Y'),
                   help='world XY to place the cube centre, instead of a tray')
    p.add_argument('--pregrasp-height', type=float, default=0.15,
                   help='TCP height above the cube centre before descending (m)')
    p.add_argument('--lift-height', type=float, default=0.10)
    p.add_argument('--speed', type=float, default=0.5,
                   help='fraction of joint velocity limits (0-1]')
    p.add_argument('--accel', type=float, default=0.5,
                   help='fraction of joint acceleration limits (0-1]')
    p.add_argument('--reset', action='store_true',
                   help='first return the cube to its initial pose in Gazebo and MoveIt')
    p.add_argument('--out', help='default: /tmp/panda_pick_place_<arm>.json')
    args = p.parse_args()

    arm = ARMS[args.arm]
    cube = args.cube or arm.default_cube
    if args.place:
        place = args.place
    else:
        # Live tray position: the scenario randomizer moves the trays.
        place = gz_model_pose(args.dropoff or arm.default_dropoff)[0][:2]
    place_xyz = (place[0], place[1], CUBES[cube][2])
    out = args.out or f'/tmp/panda_pick_place_{arm.name}.json'

    rclpy.init()
    node = PickPlace(arm, args.speed, args.accel)
    spin = Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    try:
        if args.reset:
            if args.execute:
                node.open_gripper()  # release the cube before teleporting it
            reset_cube(node, cube)
        print(f"{'Executing' if args.execute else 'Planning'} {arm.group} pick of "
              f"{cube} -> place {place_xyz}")
        # Live pose; the cube may have been moved or yawed.
        cube_xyz, cube_yaw = gz_model_pose(cube)
        stages = run(node, cube, cube_xyz, place_xyz, args.pregrasp_height,
                     args.lift_height, args.execute, cube_yaw)
        record = dict(arm=arm.group, cube=cube, cube_xyz=list(cube_xyz), cube_yaw=cube_yaw,
                      place_xyz=list(place_xyz), target_link=arm.link8,
                      tcp_offset=TCP_OFFSET, speed=args.speed, accel=args.accel,
                      executed=args.execute, **summarize(stages, args.execute),
                      stages=stages)
        Path(out).write_text(json.dumps(record, indent=2) + '\n')
        print(f'Saved {out}')
        if not record['success']:
            raise RuntimeError(f"Stage '{record['failed_stage']}' failed")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=2)


if __name__ == '__main__':
    main()
