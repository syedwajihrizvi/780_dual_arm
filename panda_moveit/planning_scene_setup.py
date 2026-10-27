
#!/usr/bin/env python3

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene


EXPECTED_CUBES = 12

WORLD_FILE = (
    Path(get_package_share_directory("panda_moveit"))
    / "worlds/panda_world.sdf"
)


def parse_pose(element):
    """Read an SDF pose as xyz + quaternion."""
    if element is None or not element.text:
        values = [0.0] * 6
    else:
        values = [float(x) for x in element.text.split()]
        values += [0.0] * (6 - len(values))

    x, y, z, roll, pitch, yaw = values[:6]

    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    q = (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )

    return (x, y, z), q


def multiply_quaternions(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b

    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate_vector(q, v):
    x, y, z, w = q
    vx, vy, vz = v

    u = (x, y, z)
    dot = x * vx + y * vy + z * vz
    norm = x*x + y*y + z*z

    cx = y*vz - z*vy
    cy = z*vx - x*vz
    cz = x*vy - y*vx

    return (
        2*dot*x + (w*w - norm)*vx + 2*w*cx,
        2*dot*y + (w*w - norm)*vy + 2*w*cy,
        2*dot*z + (w*w - norm)*vz + 2*w*cz,
    )


def compose(a, b):
    """Compose two XYZ/quaternion transforms."""
    pa, qa = a
    pb, qb = b
    rb = rotate_vector(qa, pb)

    return (
        tuple(pa[i] + rb[i] for i in range(3)),
        multiply_quaternions(qa, qb),
    )


def to_ros_pose(transform):
    position, orientation = transform

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = position

    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = orientation

    return pose


def read_world_objects():
    if not WORLD_FILE.exists():
        raise FileNotFoundError(WORLD_FILE)

    root = ET.parse(WORLD_FILE).getroot()
    world = root.find("world")

    if world is None:
        raise RuntimeError("SDF does not contain a world element")

    objects = []
    cube_count = 0
    table_count = 0

    for model in world.findall("model"):
        model_name = model.get("name", "")
        model_tf = parse_pose(model.find("pose"))

        boxes = []

        for link in model.findall("link"):
            link_tf = compose(
                model_tf,
                parse_pose(link.find("pose")),
            )

            for collision in link.findall("collision"):
                box = collision.find("geometry/box/size")

                if box is None or not box.text:
                    continue

                dimensions = [
                    float(value)
                    for value in box.text.split()
                ]

                collision_tf = compose(
                    link_tf,
                    parse_pose(collision.find("pose")),
                )

                boxes.append((dimensions, collision_tf))

        # Identify 4 cm cubes by collision dimensions,
        # rather than relying on their model names.
        is_cube = (
            len(boxes) == 1
            and all(
                0.02 <= size <= 0.08
                for size in boxes[0][0]
            )
        )

        # Identify the worktable from its large tabletop.
        is_table = any(
            max(dimensions) >= 1.0
            and sorted(dimensions)[1] >= 0.5
            for dimensions, _ in boxes
        )

        # Include our custom obstacle walls and drop-off trays.
        is_obstacle = (
            model_name.startswith(("obstacle_wall_", "dropoff_"))
            and len(boxes) > 0
        )

        if not (is_cube or is_table or is_obstacle):
            continue

        if is_cube:
            cube_count += 1

        if is_table:
            table_count += 1

        collision_object = CollisionObject()
        collision_object.header.frame_id = "world"
        collision_object.id = model_name
        collision_object.operation = CollisionObject.ADD

        for dimensions, transform in boxes:
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = dimensions

            collision_object.primitives.append(primitive)
            collision_object.primitive_poses.append(
                to_ros_pose(transform)
            )

        objects.append(collision_object)

    if cube_count != EXPECTED_CUBES or table_count != 1:
        raise RuntimeError(
            f"Expected {EXPECTED_CUBES} cubes and 1 table, found "
            f"{cube_count} cubes and {table_count} tables. "
            "Check the SDF collision definitions."
        )

    return objects


class PlanningSceneSetup(Node):

    def __init__(self):
        super().__init__("planning_scene_setup")

        self.client = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
        )

    def apply(self, objects):
        self.get_logger().info(
            "Waiting for /apply_planning_scene..."
        )

        if not self.client.wait_for_service(timeout_sec=60.0):
            raise RuntimeError(
                "MoveIt planning scene service unavailable"
            )

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.world.collision_objects = objects

        request = ApplyPlanningScene.Request()
        request.scene = scene

        future = self.client.call_async(request)

        rclpy.spin_until_future_complete(
            self, future, timeout_sec=10.0
        )

        if not future.done():
            raise TimeoutError("Planning scene request timed out")

        response = future.result()

        if response is None or not response.success:
            raise RuntimeError(
                "MoveIt rejected planning scene update"
            )

        self.get_logger().info(
            f"Successfully added {len(objects)} "
            "collision objects to MoveIt"
        )

        for obj in objects:
            self.get_logger().info(
                f"  {obj.id}: {len(obj.primitives)} box(es)"
            )


def main(args=None):
    rclpy.init(args=args)
    node = PlanningSceneSetup()

    try:
        objects = read_world_objects()
        node.apply(objects)

    except Exception as exc:
        node.get_logger().error(str(exc))

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
