
import copy
import threading
import time
from functools import partial

import rclpy
from rclpy.node import Node
from rclpy.action import (
    ActionClient,
    ActionServer,
    GoalResponse,
    CancelResponse,
)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory


JOINTS = [f"panda_joint{i}" for i in range(1, 8)]

CONTROLLERS = {
    "first": "/panda_arm_controller/follow_joint_trajectory",
    "second": "/second/panda_arm_controller/follow_joint_trajectory",
}

# Maximum additional time allowed after the final trajectory point.
EXECUTION_TIMEOUT_MARGIN = 5.0

# Maximum time to wait for a cancellation result.
CANCEL_TIMEOUT = 3.0


class DualTrajectoryAdapter(Node):

    def __init__(self):
        super().__init__("dual_trajectory_adapter")

        self.callback_group = ReentrantCallbackGroup()
        self.trajectory_clients = {}
        self.trajectory_servers = {}

        self.downstream_goals = {
            "first": None,
            "second": None,
        }

        self.active_goals = {
            "first": None,
            "second": None,
        }

        self.cancel_pending = {
            "first": False,
            "second": False,
        }

        self.goal_lock = threading.RLock()

        for arm, target in CONTROLLERS.items():

            self.trajectory_clients[arm] = ActionClient(
                self,
                FollowJointTrajectory,
                target,
                callback_group=self.callback_group,
            )

            self.trajectory_servers[arm] = ActionServer(
                self,
                FollowJointTrajectory,
                f"/{arm}/follow_joint_trajectory",
                goal_callback=partial(self.check_goal, arm),
                cancel_callback=partial(self.cancel_goal, arm),
                handle_accepted_callback=partial(
                    self.handle_accepted, arm
                ),
                execute_callback=partial(self.execute, arm),
                callback_group=self.callback_group,
            )

        self.get_logger().info(
            "Dual Panda trajectory adapter ready"
        )

    # --------------------------------------------------
    # Validate incoming MoveIt trajectory
    # --------------------------------------------------

    def check_goal(self, arm, request):

        expected = {
            f"{arm}_{name}" for name in JOINTS
        }

        actual = request.trajectory.joint_names

        if (
            len(actual) != 7
            or set(actual) != expected
            or not request.trajectory.points
        ):
            self.get_logger().error(
                f"Invalid joint names or empty trajectory "
                f"for {arm}"
            )
            return GoalResponse.REJECT

        # Reject malformed trajectory points.
        previous_time = -1.0

        for point in request.trajectory.points:

            if len(point.positions) != 7:
                self.get_logger().error(
                    f"Invalid trajectory positions for {arm}"
                )
                return GoalResponse.REJECT

            for values in (
                point.velocities,
                point.accelerations,
                point.effort,
            ):
                if values and len(values) != 7:
                    self.get_logger().error(
                        f"Invalid trajectory array for {arm}"
                    )
                    return GoalResponse.REJECT

            t = (
                point.time_from_start.sec
                + point.time_from_start.nanosec * 1e-9
            )

            if t < 0 or t <= previous_time:
                self.get_logger().error(
                    f"Invalid trajectory timing for {arm}"
                )
                return GoalResponse.REJECT

            previous_time = t

        with self.goal_lock:
            if self.active_goals[arm] is not None:
                self.get_logger().warn(
                    f"{arm} Panda is already executing"
                )
                return GoalResponse.REJECT

        if not self.trajectory_clients[arm].server_is_ready():
            self.get_logger().error(
                f"Gazebo controller for {arm} is unavailable"
            )
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    # --------------------------------------------------
    # Reserve arm before beginning execution
    # --------------------------------------------------

    def handle_accepted(self, arm, handle):

        with self.goal_lock:
            if self.active_goals[arm] is not None:
                self.get_logger().error(
                    f"Overlapping goal received for {arm}"
                )
                handle.abort()
                return

            self.active_goals[arm] = handle
            self.cancel_pending[arm] = False

        handle.execute()

    # --------------------------------------------------
    # Handle cancellation requests
    # --------------------------------------------------

    def cancel_goal(self, arm, handle):

        with self.goal_lock:
            if self.active_goals[arm] is not handle:
                return CancelResponse.REJECT

            self.cancel_pending[arm] = True
            downstream = self.downstream_goals[arm]

        self.get_logger().warn(
            f"Cancellation requested for {arm} Panda"
        )

        if downstream is not None:
            downstream.cancel_goal_async()

        return CancelResponse.ACCEPT

    # --------------------------------------------------
    # Execute trajectory
    # --------------------------------------------------

    async def execute(self, arm, handle):

        client = self.trajectory_clients[arm]

        result = FollowJointTrajectory.Result()

        try:
            # Copy incoming MoveIt trajectory.
            command = copy.deepcopy(handle.request)

            # Convert prefixed names to Gazebo joint names.
            command.trajectory.joint_names = [
                name.removeprefix(f"{arm}_")
                for name in command.trajectory.joint_names
            ]

            # Convert optional path tolerances.
            for tolerance in command.path_tolerance:
                tolerance.name = tolerance.name.removeprefix(
                    f"{arm}_"
                )

            # Convert optional goal tolerances.
            for tolerance in command.goal_tolerance:
                tolerance.name = tolerance.name.removeprefix(
                    f"{arm}_"
                )

            duration = (
                command.trajectory.points[-1].time_from_start.sec
                + command.trajectory.points[-1].time_from_start.nanosec
                * 1e-9
            )

            timeout = duration + EXECUTION_TIMEOUT_MARGIN

            self.get_logger().info(
                f"Forwarding {arm} Panda trajectory "
                f"({duration:.3f}s)"
            )

            # Send goal to the correct Gazebo controller.
            send_future = client.send_goal_async(
                command,
                feedback_callback=partial(
                    self.forward_feedback, arm, handle
                ),
            )

            client_goal = await send_future

            if not client_goal.accepted:
                handle.abort()
                result.error_code = result.INVALID_GOAL
                result.error_string = (
                    "Gazebo controller rejected trajectory"
                )
                return result

            with self.goal_lock:
                self.downstream_goals[arm] = client_goal
                cancel_requested = self.cancel_pending[arm]

            if cancel_requested:
                client_goal.cancel_goal_async()

            result_future = client_goal.get_result_async()
            started = time.monotonic()
            cancel_started = None
            cancellation_sent = cancel_requested

            # Monitor completion, timeout and cancellation.
            while not result_future.done():

                with self.goal_lock:
                    cancel_requested = self.cancel_pending[arm]

                if cancel_requested:

                    if not cancellation_sent:
                        client_goal.cancel_goal_async()
                        cancellation_sent = True

                    if cancel_started is None:
                        cancel_started = time.monotonic()

                    if (
                        time.monotonic() - cancel_started
                        > CANCEL_TIMEOUT
                    ):
                        self.get_logger().error(
                            f"{arm} cancellation timed out"
                        )
                        handle.abort()
                        result.error_code = result.INVALID_GOAL
                        result.error_string = (
                            "Cancellation timed out; "
                            "downstream goal may still be active"
                        )
                        return result

                elif time.monotonic() - started > timeout:

                    self.get_logger().error(
                        f"{arm} trajectory timed out"
                    )

                    client_goal.cancel_goal_async()

                    with self.goal_lock:
                        self.cancel_pending[arm] = True

                    cancellation_sent = True
                    cancel_started = time.monotonic()

                await self.ros_sleep(0.02)

            response = result_future.result()
            result = response.result

            if response.status == GoalStatus.STATUS_CANCELED:
                handle.canceled()
                self.get_logger().warn(
                    f"{arm} Panda trajectory canceled"
                )

            elif (
                response.status == GoalStatus.STATUS_SUCCEEDED
                and result.error_code == result.SUCCESSFUL
            ):
                handle.succeed()
                self.get_logger().info(
                    f"{arm} Panda trajectory succeeded"
                )

            else:
                handle.abort()
                self.get_logger().error(
                    f"{arm} Panda execution failed: "
                    f"status={response.status}, "
                    f"error={result.error_code}, "
                    f"message={result.error_string}"
                )

            return result

        except Exception as exc:

            self.get_logger().error(
                f"{arm} trajectory exception: {exc}"
            )

            if handle.is_active:
                handle.abort()

            result.error_code = result.INVALID_GOAL
            result.error_string = str(exc)

            return result

        finally:

            with self.goal_lock:
                self.downstream_goals[arm] = None
                self.active_goals[arm] = None
                self.cancel_pending[arm] = False

    # --------------------------------------------------
    # Forward controller feedback to MoveIt
    # --------------------------------------------------

    def forward_feedback(self, arm, handle, msg):

        if not handle.is_active:
            return

        feedback = copy.deepcopy(msg.feedback)

        feedback.joint_names = [
            f"{arm}_{name}"
            for name in feedback.joint_names
        ]

        handle.publish_feedback(feedback)

    async def ros_sleep(self, seconds):
        future = rclpy.task.Future()

        timer = self.create_timer(
            seconds,
            lambda: future.set_result(True) if not future.done() else None,
            callback_group=self.callback_group,
        )

        try:
            await future
        finally:
            self.destroy_timer(timer)


def main(args=None):

    rclpy.init(args=args)

    node = DualTrajectoryAdapter()

    executor = MultiThreadedExecutor(
        num_threads=4
    )

    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        executor.shutdown()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
