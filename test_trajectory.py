
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class PandaTrajectoryTest(Node):

    def __init__(self):
        super().__init__('panda_trajectory_test')

        self.joint_names = [
            f'panda_joint{i}' for i in range(1, 8)
        ]

        self.current_positions = None

        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        self.client = ActionClient(
            self,
            FollowJointTrajectory,
            '/panda_arm_controller/follow_joint_trajectory'
        )

    def joint_callback(self, msg):
        positions = dict(zip(msg.name, msg.position))

        if all(name in positions for name in self.joint_names):
            values = [
                positions[name] for name in self.joint_names
            ]

            if all(math.isfinite(v) for v in values):
                self.current_positions = values

    def send_goal(self):
        self.get_logger().info('Waiting for joint states...')

        deadline = self.get_clock().now().nanoseconds + 5_000_000_000

        while rclpy.ok() and self.current_positions is None:
            rclpy.spin_once(self, timeout_sec=0.1)

            if self.get_clock().now().nanoseconds > deadline:
                self.get_logger().error('No valid joint states received')
                return

        target = self.current_positions.copy()

        # Small movement of joint 1
        target[0] += 0.05

        self.get_logger().info(
            f'Current positions: {self.current_positions}'
        )

        self.get_logger().info(
            f'Target positions: {target}'
        )

        if not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Trajectory action unavailable')
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names

        start = JointTrajectoryPoint()
        start.positions = self.current_positions
        start.time_from_start = Duration(sec=0)

        end = JointTrajectoryPoint()
        end.positions = target
        end.time_from_start = Duration(sec=5)

        goal.trajectory.points = [start, end]

        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        handle = future.result()

        if handle is None or not handle.accepted:
            self.get_logger().error('Trajectory rejected')
            return

        self.get_logger().info('Trajectory accepted')

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        response = result_future.result()

        if response is None:
            self.get_logger().error('No trajectory result received')
            return

        self.get_logger().info(
            f'Result error code: {response.result.error_code}'
        )


def main():
    rclpy.init()
    node = PandaTrajectoryTest()

    try:
        node.send_goal()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

