
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class DualJointStates(Node):
    def __init__(self):
        super().__init__("dual_joint_states")

        self.publisher = self.create_publisher(
            JointState,
            "/dual_joint_states",
            10,
        )

        self.first_states = {}
        self.second_states = {}

        self.create_subscription(
            JointState,
            "/joint_states",
            self.first_callback,
            10,
        )

        self.create_subscription(
            JointState,
            "/second/joint_states",
            self.second_callback,
            10,
        )

        self.create_timer(0.02, self.publish_states)

        self.get_logger().info(
            "Dual Panda joint-state merger started"
        )

    def update_states(self, msg, storage):
        for i, name in enumerate(msg.name):
            position = (
                msg.position[i]
                if i < len(msg.position)
                else 0.0
            )
            velocity = (
                msg.velocity[i]
                if i < len(msg.velocity)
                else 0.0
            )
            effort = (
                msg.effort[i]
                if i < len(msg.effort)
                else 0.0
            )

            storage[name] = (
                position,
                velocity,
                effort,
            )

    def first_callback(self, msg):
        self.update_states(msg, self.first_states)

    def second_callback(self, msg):
        self.update_states(msg, self.second_states)

    def publish_states(self):
        if not self.first_states or not self.second_states:
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()

        for prefix, states in (
            ("first_", self.first_states),
            ("second_", self.second_states),
        ):
            for name, values in sorted(states.items()):
                msg.name.append(prefix + name)
                msg.position.append(values[0])
                msg.velocity.append(values[1])
                msg.effort.append(values[2])

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DualJointStates()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
