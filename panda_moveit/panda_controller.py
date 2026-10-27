
#!/usr/bin/env python3

import argparse
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, Pose
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_pose

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    MoveItErrorCodes,
)
from shape_msgs.msg import SolidPrimitive


BASE_FRAME = "panda_link0"
EE_LINK = "panda_link8"


class PandaController(Node):
    def __init__(self):
        super().__init__("panda_controller")

        self.client = ActionClient(
            self, MoveGroup, "/move_action"
        )

        # Store available transformations.
        self.tf_buffer = Buffer()

        # Listen to /tf and /tf_static.
        self.tf_listener = TransformListener(
            self.tf_buffer, self
        )

    def transform_pose(self, values, source_frame):
        """
        Convert a target pose from source_frame
        into panda_link0 using TF2.
        """
        x, y, z, qx, qy, qz, qw = values

        norm = math.sqrt(
            qx*qx + qy*qy + qz*qz + qw*qw
        )
        if norm < 1e-9:
            raise ValueError("Invalid zero quaternion.")

        target = PoseStamped()
        target.header.frame_id = source_frame

        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.position.z = z

        target.pose.orientation.x = qx / norm
        target.pose.orientation.y = qy / norm
        target.pose.orientation.z = qz / norm
        target.pose.orientation.w = qw / norm

        if source_frame == BASE_FRAME:
            return target

        self.get_logger().info(
            f"Transforming {source_frame} -> {BASE_FRAME}"
        )

        # Wait for the listener to receive TF messages.
        deadline = time.monotonic() + 5.0

        while rclpy.ok():
            try:
                transform = self.tf_buffer.lookup_transform(
                    BASE_FRAME,
                    source_frame,
                    Time(),  # Latest available transform
                )

                converted_pose = do_transform_pose(
                    target.pose, transform
                )
                converted = PoseStamped()

                converted.header.frame_id = BASE_FRAME
                converted.header.stamp = transform.header.stamp
                converted.pose = converted_pose

                p = converted.pose.position
                q = converted.pose.orientation

                self.get_logger().info(
                    f"Position in {BASE_FRAME}: "
                    f"[{p.x:.4f}, {p.y:.4f}, {p.z:.4f}]"
                )
                self.get_logger().info(
                    f"Orientation in {BASE_FRAME}: "
                    f"[{q.x:.4f}, {q.y:.4f}, "
                    f"{q.z:.4f}, {q.w:.4f}]"
                )

                return converted

            except TransformException as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Cannot transform {source_frame} "
                        f"to {BASE_FRAME}: {exc}"
                    ) from exc

                rclpy.spin_once(self, timeout_sec=0.1)

        raise RuntimeError("ROS shutdown during TF lookup.")

    def create_pose_goal(self, pose_stamped):
        """
        Create MoveIt goal constraints.
        pose_stamped is already in panda_link0.
        """
        pose = pose_stamped.pose

        constraints = Constraints()
        constraints.name = "panda_pose_goal"

        position = PositionConstraint()
        position.header.frame_id = BASE_FRAME
        position.link_name = EE_LINK
        position.weight = 1.0

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.02, 0.02, 0.02]

        # The region box is centered at the target point.
        region_pose = Pose()
        region_pose.position = pose.position
        region_pose.orientation.w = 1.0

        position.constraint_region.primitives.append(box)
        position.constraint_region.primitive_poses.append(
            region_pose
        )

        orientation = OrientationConstraint()
        orientation.header.frame_id = BASE_FRAME
        orientation.link_name = EE_LINK
        orientation.orientation = pose.orientation

        orientation.absolute_x_axis_tolerance = 0.15
        orientation.absolute_y_axis_tolerance = 0.15
        orientation.absolute_z_axis_tolerance = 0.15
        orientation.weight = 1.0

        constraints.position_constraints.append(position)
        constraints.orientation_constraints.append(
            orientation
        )

        return constraints

    def move_to_pose(self, values, frame, max_attempts=5):
        # First convert the target into the robot base frame.
        try:
            target = self.transform_pose(values, frame)
        except (RuntimeError, ValueError) as exc:
            self.get_logger().error(str(exc))
            return False

        constraints = self.create_pose_goal(target)

        if not self.client.wait_for_server(
            timeout_sec=10.0
        ):
            self.get_logger().error(
                "MoveIt /move_action unavailable."
            )
            return False

        retryable = {
            MoveItErrorCodes.PLANNING_FAILED,
            MoveItErrorCodes.TIMED_OUT,
            MoveItErrorCodes.GOAL_IN_COLLISION,
            MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED,
            MoveItErrorCodes.NO_IK_SOLUTION,
        }

        for attempt in range(1, max_attempts + 1):
            self.get_logger().info(
                f"Planning attempt {attempt}/{max_attempts}"
            )

            goal = MoveGroup.Goal()
            request = goal.request

            request.group_name = "panda_arm"
            request.num_planning_attempts = 1
            request.allowed_planning_time = 3.0
            request.max_velocity_scaling_factor = 0.1
            request.max_acceleration_scaling_factor = 0.1

            # Use the current robot state.
            request.start_state.is_diff = True
            request.goal_constraints = [constraints]

            # Plan and execute in one request.
            goal.planning_options.plan_only = False
            goal.planning_options.look_around = False
            goal.planning_options.replan = False

            goal.planning_options.planning_scene_diff.is_diff = True
            goal.planning_options.planning_scene_diff.robot_state.is_diff = True

            send_future = self.client.send_goal_async(goal)
            rclpy.spin_until_future_complete(
                self, send_future
            )

            handle = send_future.result()

            if handle is None or not handle.accepted:
                self.get_logger().error(
                    "MoveIt rejected the goal."
                )
                return False

            result_future = handle.get_result_async()
            rclpy.spin_until_future_complete(
                self, result_future
            )

            response = result_future.result()
            if response is None:
                self.get_logger().error(
                    "No result from MoveIt."
                )
                return False

            result = response.result
            error = result.error_code.val

            if error == MoveItErrorCodes.SUCCESS:
                self.get_logger().info(
                    f"SUCCESS after {attempt} attempt(s)."
                )
                self.get_logger().info(
                    f"Planning time: "
                    f"{result.planning_time:.3f}s"
                )
                return True

            self.get_logger().warning(
                f"Attempt {attempt} failed "
                f"(MoveIt error: {error})"
            )

            if error not in retryable:
                self.get_logger().error(
                    f"Motion aborted after "
                    f"{attempt} attempt(s)."
                )
                return False

        self.get_logger().error(
            f"Cannot reach pose after "
            f"{max_attempts} planning attempts."
        )
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Move Panda to a pose in any TF frame."
    )

    parser.add_argument(
        "pose",
        nargs=7,
        type=float,
        metavar="VALUE",
        help="X Y Z QX QY QZ QW",
    )

    parser.add_argument(
        "--frame",
        default=BASE_FRAME,
        help="Source TF frame (default: panda_link0)",
    )

    parser.add_argument(
        "--attempts",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if args.attempts < 1:
        parser.error("--attempts must be >= 1")

    rclpy.init()
    node = PandaController()

    try:
        success = node.move_to_pose(
            args.pose,
            args.frame,
            args.attempts,
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
