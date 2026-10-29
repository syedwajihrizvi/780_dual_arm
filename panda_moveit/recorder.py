#!/usr/bin/env python3
"""Records executed runs at the overhead camera's rate (10 Hz): RGB video,
depth, both arms' joint states, the gripper commands and the target cube's
pose, one row per RGB frame.

Used by collect.py --execute; each execution gets its own folder:
    rgb.mp4      960x720 RGB video (mp4v), one frame per row
    depth.npz    'depth_mm' uint16 (n, 360, 480), millimetres, 0 = no return
    frames.npz   per row: t (s since start), sim_t (s), and per arm
                 <arm>_position / <arm>_velocity (n, 9: joints 1-7, fingers 1-2),
                 <arm>_gripper_command (n,; NaN before the first command),
                 cube_pose (n, 4: x, y, z, yaw)

The recording runs in its own process. gz-transport delivers topic messages
and service replies on one thread, so in-process Python subscriptions that
wait for the GIL stall the gz requests (set_pose, remove, ...) that the
main process makes, until they time out.
"""
import math
import multiprocessing
import os
from functools import partial
from pathlib import Path
from threading import Lock, Thread
import time

import numpy as np

CAMERA_TOPIC = '/overhead_camera'
FPS = 10
DEPTH_SIZE = (480, 360)  # width, height; half the camera resolution
JOINTS = [f'panda_joint{i}' for i in range(1, 8)] + ['panda_finger_joint1', 'panda_finger_joint2']
TIMEOUT = 20.0  # s to wait for the recording process to answer


class Recorder:
    """Main-process handle; the recording happens in a child process."""

    def __init__(self, nodes):
        """nodes: {arm name: PickPlace node}, for their controller namespaces."""
        namespaces = {arm: node.arm.ns for arm, node in nodes.items()}
        context = multiprocessing.get_context('spawn')  # no forked rclpy state
        self.conn, child = context.Pipe()
        self.process = context.Process(target=_serve, args=(child, namespaces), daemon=True)
        self.process.start()
        self._call('ready')

    def _call(self, *message):
        self.conn.send(message)
        if not self.conn.poll(TIMEOUT):
            raise RuntimeError(f'recorder did not answer {message[0]}')
        reply = self.conn.recv()
        if isinstance(reply, Exception):
            raise reply
        return reply

    def start(self, folder, cube):
        """Begin recording into folder, tracking the pose of model cube.
        Returns the start time (time.monotonic(), shared across processes)."""
        return self._call('start', str(folder), cube)

    def stop(self):
        """Finish the recording; returns a summary for the trial record."""
        return self._call('stop')

    def close(self):
        if self.process.is_alive():
            self.conn.send(('quit',))
            self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()


def _serve(conn, namespaces):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    rclpy.init()
    worker = _Worker(namespaces)
    executor = SingleThreadedExecutor()
    executor.add_node(worker.node)
    Thread(target=executor.spin, daemon=True).start()
    while True:
        try:
            command, *args = conn.recv()
        except EOFError:
            command = 'quit'  # the main process is gone
        if command == 'quit':
            executor.shutdown()
            worker.close()
            rclpy.shutdown()
            # gz-transport's threads abort ('terminate called ...') if left to
            # the interpreter's teardown; everything is saved by now.
            os._exit(0)
        try:
            conn.send(getattr(worker, command)(*args))
        except Exception as e:
            conn.send(RuntimeError(f'recorder {command}: {type(e).__name__}: {e}'))


def _stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9


class _Worker:
    def __init__(self, namespaces):
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
        from gz.transport13 import Node as GzNode
        from gz.msgs10.image_pb2 import Image
        from gz.msgs10.pose_v_pb2 import Pose_V
        from pick_place import GZ_WORLD

        self.lock = Lock()
        self.active = False
        self.joints = {arm: None for arm in namespaces}
        self.commands = {arm: math.nan for arm in namespaces}
        self.cube, self.cube_pose = None, None
        self.node = Node('execution_recorder')
        for arm, ns in namespaces.items():
            self.node.create_subscription(JointState, f'{ns}/joint_states',
                                          partial(self.on_joints, arm), 10)
            self.node.create_subscription(Float64MultiArray, f'{ns}/panda_gripper_controller/commands',
                                          partial(self.on_command, arm), 10)
        self.gz = GzNode()
        self.gz.subscribe(Image, f'{CAMERA_TOPIC}/image', self.on_rgb)
        self.gz.subscribe(Image, f'{CAMERA_TOPIC}/depth_image', self.on_depth)
        self.pose_topic = f'/world/{GZ_WORLD}/pose/info'
        self.gz.subscribe(Pose_V, self.pose_topic, self.on_poses)

    # -- commands from the main process -----------------------------

    def ready(self):
        deadline = time.monotonic() + TIMEOUT / 2
        while any(j is None for j in self.joints.values()):
            if time.monotonic() > deadline:
                raise TimeoutError(f'no joint states for {[a for a, j in self.joints.items() if j is None]}')
            time.sleep(0.05)
        return True

    def start(self, folder, cube):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.folder, self.cube, self.cube_pose = folder, cube, None
            self.video, self.rows, self.depth = None, [], {}
            self.t0 = time.monotonic()
            self.active = True
        return self.t0

    def close(self):
        for topic in (f'{CAMERA_TOPIC}/image', f'{CAMERA_TOPIC}/depth_image', self.pose_topic):
            self.gz.unsubscribe(topic)
        self.node.destroy_node()

    def stop(self):
        with self.lock:
            self.active = False
            if self.video is not None:
                self.video.release()
            rows, depth = self.rows, self.depth
        if not rows:
            return dict(frames=0)
        # RGB and depth come from one sensor with the same sim stamps; match
        # each RGB row to the nearest depth frame.
        stamps = np.array(sorted(depth))
        sim_t = np.array([r['sim_t'] for r in rows])
        if len(stamps):
            nearest = stamps[np.abs(stamps[None] - sim_t[:, None]).argmin(1)]
            depth_mm = np.stack([depth[s] for s in nearest])
            matched = float(np.abs(nearest - sim_t).max()) < 1e-3
        else:
            depth_mm, matched = np.zeros((0, DEPTH_SIZE[1], DEPTH_SIZE[0]), np.uint16), False
        np.savez_compressed(self.folder / 'depth.npz', depth_mm=depth_mm)
        np.savez_compressed(self.folder / 'frames.npz',
                            **{k: np.array([r[k] for r in rows], float) for k in rows[0]})
        return dict(frames=len(rows), fps=FPS, duration=rows[-1]['t'], depth_matched=matched)

    # -- callbacks (ROS executor and gz transport threads) ------------

    def on_joints(self, arm, msg):
        pos, vel = dict(zip(msg.name, msg.position)), dict(zip(msg.name, msg.velocity))
        if all(j in pos for j in JOINTS):
            self.joints[arm] = ([pos[j] for j in JOINTS], [vel.get(j, math.nan) for j in JOINTS])

    def on_command(self, arm, msg):
        self.commands[arm] = msg.data[0]

    def on_poses(self, msg):
        if not self.active:
            return
        for p in msg.pose:
            if p.name == self.cube:
                q = p.orientation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                self.cube_pose = [p.position.x, p.position.y, p.position.z, yaw]
                return

    def on_rgb(self, msg):
        import cv2
        with self.lock:
            if not self.active:
                return
            rgb = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
            if self.video is None:
                self.video = cv2.VideoWriter(str(self.folder / 'rgb.mp4'),
                                             cv2.VideoWriter_fourcc(*'mp4v'), FPS,
                                             (msg.width, msg.height))
            self.video.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            row = dict(t=time.monotonic() - self.t0, sim_t=_stamp(msg),
                       cube_pose=self.cube_pose or [math.nan] * 4)
            for arm, (position, velocity) in self.joints.items():
                row[f'{arm}_position'], row[f'{arm}_velocity'] = position, velocity
                row[f'{arm}_gripper_command'] = self.commands[arm]
            self.rows.append(row)

    def on_depth(self, msg):
        import cv2
        if not self.active:
            return
        depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        depth = cv2.resize(depth, DEPTH_SIZE, interpolation=cv2.INTER_NEAREST)
        mm = np.where(np.isfinite(depth), np.round(depth * 1000), 0).clip(0, 65535).astype(np.uint16)
        with self.lock:
            if self.active:
                self.depth[_stamp(msg)] = mm
