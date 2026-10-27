import os
import yaml
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import xacro

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():

    pkg = Path(get_package_share_directory("panda_moveit"))
    gz_pkg = Path(get_package_share_directory("ros_gz_sim"))
    moveit_pkg = Path(
        get_package_share_directory(
            "moveit_resources_panda_moveit_config"
        )
    )

    world = pkg / "worlds" / "panda_world.sdf"
    urdf = pkg / "config" / "urdf" / "panda.urdf.xacro"
    controllers = pkg / "config" / "moveit_controllers.yaml"

    # Generate robot description from our Xacro
    robot_description = xacro.process_file(str(urdf)).toxml()

    
    # Create a Gazebo SDF from the Panda URDF.
    # Keep the original URDF for MoveIt / robot_state_publisher.

    spawn_sdf = "/tmp/panda_fixed.sdf"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", delete=False
    ) as f:
        f.write(robot_description)
        urdf_path = f.name

    try:
        result = subprocess.run(
            ["gz", "sdf", "-p", urdf_path],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        os.unlink(urdf_path)

    # Modify the SDF, not the URDF.
    root = ET.fromstring(result.stdout)
    model = root.find("model")

    if model is None:
        raise RuntimeError("Converted SDF has no model")

    if model.find("./link[@name='panda_link0']") is None:
        raise RuntimeError("Cannot find Panda base link")

    # Anchor panda_link0 to Gazebo world.
    joint = ET.SubElement(
        model,
        "joint",
        {
            "name": "world_to_panda_base",
            "type": "fixed",
        },
    )
    ET.SubElement(joint, "parent").text = "world"
    ET.SubElement(joint, "child").text = "panda_link0"

    # Ensure the ros2_control Gazebo plugin survived conversion.
    if not any(
        "gz_ros2_control" in ET.tostring(
            plugin, encoding="unicode"
        )
        for plugin in model.findall("plugin")
    ):
        raise RuntimeError("gz_ros2_control plugin missing from SDF")

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(
        spawn_sdf,
        encoding="unicode",
        xml_declaration=True,
    )

    print(f"Generated fixed-base Panda SDF: {spawn_sdf}")

    # --------------------------------------------
    # Generate the second, independent Panda
    # --------------------------------------------

    second_sdf = "/tmp/panda_second_fixed.sdf"

    # Read the already generated and anchored SDF.
    second_tree = ET.parse(spawn_sdf)
    second_root = second_tree.getroot()
    second_model = second_root.find("model")

    if second_model is None:
        raise RuntimeError("Panda 2 SDF model not found")

    second_model.set("name", "panda_second")

    # Find the Gazebo ros2_control plugin.
    control_plugin = None

    for plugin in second_model.findall("plugin"):
        if "gz_ros2_control" in (
            plugin.get("filename", "")
            + plugin.get("name", "")
        ):
            control_plugin = plugin
            break

    if control_plugin is None:
        raise RuntimeError("Panda 2 control plugin not found")

    # Give Panda 2 a separate controller namespace.
    ros = control_plugin.find("ros")

    if ros is None:
        ros = ET.SubElement(control_plugin, "ros")

    namespace = ros.find("namespace")

    if namespace is None:
        namespace = ET.SubElement(ros, "namespace")

    namespace.text = "/second"

    # Subscribe to Panda 2's robot description.
    param_node = control_plugin.find("robot_param_node")

    if param_node is None:
        param_node = ET.SubElement(
            control_plugin, "robot_param_node"
        )

    param_node.text = "robot_state_publisher"

    # Give Panda 2 its own controller parameters.
    second_params = str(
        pkg / "config" / "ros2_controllers_second.yaml"
    )

    parameters = control_plugin.find("parameters")

    if parameters is None:
        parameters = ET.SubElement(
            control_plugin, "parameters"
        )

    parameters.text = second_params

    ET.indent(second_tree, space="  ")

    second_tree.write(
        second_sdf,
        encoding="unicode",
        xml_declaration=True,
    )

    print(f"Generated Panda 2 SDF: {second_sdf}")


    # MoveIt planning configuration
    moveit_config = (
        MoveItConfigsBuilder(
            "panda",
            package_name="moveit_resources_panda_moveit_config",
        )
        .robot_description(file_path=str(urdf))
        .trajectory_execution(file_path=str(controllers))
        .to_moveit_configs()
    )

    # --------------------------------------------
    # Dual-Panda MoveIt configuration
    # --------------------------------------------

    dual_urdf = pkg / "config" / "urdf" / "dual_panda.urdf"
    dual_srdf = pkg / "config" / "dual_panda.srdf"
    dual_kinematics = pkg / "config" / "dual_kinematics.yaml"

    with open(dual_kinematics) as f:
        kinematics_config = yaml.safe_load(f)

    dual_robot_params = {
        "robot_description": dual_urdf.read_text(),
        "robot_description_semantic": dual_srdf.read_text(),
        "robot_description_kinematics": kinematics_config,
    }

    # Reuse the existing MoveIt planning pipeline.
    dual_moveit_params = moveit_config.to_dict().copy()

    # Replace the single-Panda descriptions.
    dual_moveit_params.update(dual_robot_params)

    # The existing joint-limit overrides refer to unprefixed
    # single-Panda joints. Use the limits from dual_panda.urdf.
    dual_moveit_params.pop("robot_description_planning", None)
    dual_joint_limits = {}

    # Franka Panda datasheet joint acceleration limits (rad/s^2).
    # Scripts slow motion down with velocity/acceleration scaling factors.
    panda_max_acceleration = [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]

    for arm in ("first", "second"):
        for joint in range(1, 8):
            joint_name = f"{arm}_panda_joint{joint}"

            dual_joint_limits[joint_name] = {
                "has_acceleration_limits": True,
                "max_acceleration": panda_max_acceleration[joint - 1],
            }

    dual_moveit_params["robot_description_planning"] = {
        "joint_limits": dual_joint_limits
    }

    # Disable execution until both planning groups work.
    dual_moveit_params["allow_trajectory_execution"] = True

    # Use OMPL only for initial dual-arm planning.
    # Pilz requires Cartesian limits that we have not configured.
    dual_moveit_params["planning_pipelines"] = ["ompl"]
    dual_moveit_params["default_planning_pipeline"] = "ompl"


    # ---------------------------------------------------------
    # OMPL planner configurations for both Panda arms
    # ---------------------------------------------------------

    ompl_params = dual_moveit_params["ompl"]

    planner_configs = ompl_params.setdefault("planner_configs", {})

    # Fast, feasible-path planner (existing baseline)
    planner_configs["RRTConnectkConfigDefault"] = {
        "type": "geometric::RRTConnect",
        "range": 0.0,
    }

    # Asymptotically optimal path planner
    planner_configs["RRTstarkConfigDefault"] = {
        "type": "geometric::RRTstar",
        "range": 0.0,
        "goal_bias": 0.05,
        "delay_collision_checking": 1,
        "optimization_objective": "PathLengthOptimizationObjective",
    }

    # Expose both planners to each planning group.
    for group in (
        "first_panda_arm",
        "second_panda_arm",
        "both_panda_arms",
    ):
        ompl_params[group] = {
            "planner_configs": [
                "RRTConnectkConfigDefault",
                "RRTstarkConfigDefault",
            ],
            "default_planner_config": "RRTConnectkConfigDefault",
            "longest_valid_segment_fraction": 0.005,
        }


    # Remove other planning pipeline configurations.
    for pipeline in ("pilz_industrial_motion_planner", "chomp", "stomp"):
        dual_moveit_params.pop(pipeline, None)

    dual_controllers = pkg / "config" / "dual_moveit_controllers.yaml"

    with open(dual_controllers) as f:
        dual_controller_params = yaml.safe_load(f)

    dual_moveit_params.update(dual_controller_params)

    # 1. Start Gazebo with our saved Panda world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(gz_pkg / "launch" / "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": f"-r -v 3 {world}",
        }.items(),
    )

    # 2. Publish the Panda robot description
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {"robot_description": robot_description},
            {"frame_prefix": "first_"},
            {"use_sim_time": True},
        ],
    )
    second_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="second",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {"robot_description": robot_description},
            {"frame_prefix": "second_"},
            {"use_sim_time": True},
        ],
    )

    # 3. Bridge the simulation clock
    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/world/panda_world/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "--ros-args",
            "-r",
            "/world/panda_world/clock:=/clock",
        ],
        output="screen",
    )

    # 4. Spawn our customized Panda
    spawn_panda = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-world", "panda_world",
            "-file", spawn_sdf,
            "-name", "panda",
            "-x", "0.2",
            "-y", "0.0",
            "-z", "1.025",
        ],
        output="screen",
    )

    spawn_second_panda = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-world", "panda_world",
            "-file", second_sdf,
            "-name", "panda_second",
            "-x", "1.50",
            "-y", "0.0",
            "-z", "1.025",
            "-Y", "3.141592653589793",
        ],
        output="screen",
    )

    # 5. Activate joint-state broadcaster
    joint_state_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "120",
        ],
        output="screen",
    )

    # 6. Activate Panda trajectory controller
    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "panda_arm_controller",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "120",
        ],
        output="screen",
    )


    second_controllers_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "panda_arm_controller",
            "--controller-manager",
            "/second/controller_manager",
            "--controller-manager-timeout",
            "120",
            "--service-call-timeout",
            "60",
            "--activate-as-group",
        ],
        output="screen",
    )

    start_second_controllers = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_second_panda,
            on_exit=[
                TimerAction(
                    period=2.0,
                    actions=[second_controllers_spawner],
                )
            ],
        )
    )


    # 7. Start MoveIt
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            dual_moveit_params,
            {"use_sim_time": True},
        ],
        remappings=[
            ("/joint_states", "/dual_joint_states"),
        ],
    )

    # Locate RViz configuration
    rviz_candidates = [
        moveit_pkg / "config" / "moveit.rviz",
        moveit_pkg / "launch" / "moveit.rviz",
    ]
    rviz_file = next(
        (p for p in rviz_candidates if p.exists()),
        None,
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=(
            ["-d", str(rviz_file)]
            if rviz_file else []
        ),
        parameters=[
            dual_robot_params,
            moveit_config.planning_pipelines,
            {"use_sim_time": True},
        ],
        remappings=[
            ("/joint_states", "/dual_joint_states"),
        ],
        output="screen",
    )

    # Wait for spawn command before starting controllers
    start_controllers = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_panda,
            on_exit=[
                TimerAction(
                    period=2.0,
                    actions=[joint_state_spawner],
                )
            ],
        )
    )

    # Start arm controller after broadcaster
    start_arm_controller = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_spawner,
            on_exit=[arm_controller_spawner],
        )
    )

    # Start MoveIt after controller startup attempt
    start_moveit = RegisterEventHandler(
        OnProcessExit(
            target_action=arm_controller_spawner,
            on_exit=[move_group, rviz],
        )
    )

    # 8. Publish static transform from world to panda base frame
    world_to_base_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_panda_tf",
        arguments=[
            "--x", "0.2",
            "--y", "0.0",
            "--z", "1.025",
            "--roll", "0",
            "--pitch", "0",
            "--yaw", "0",
            "--frame-id", "world",
            "--child-frame-id", "first_panda_link0",
        ],
        output="screen",
    )

    world_to_second_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_second_panda_tf",
        arguments=[
            "--x", "1.50",
            "--y", "0.0",
            "--z", "1.025",
            "--roll", "0",
            "--pitch", "0",
            "--yaw", "3.141592653589793",
            "--frame-id", "world",
            "--child-frame-id", "second_panda_link0",
        ],
        output="screen",
    )

    dual_joint_states = Node(
        package="panda_moveit",
        executable="dual_joint_states",
        name="dual_joint_states",
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    dual_trajectory_adapter_node = Node(
        package="panda_moveit",
        executable="dual_trajectory_adapter",
        name="dual_trajectory_adapter",
        output="screen",
    )

    planning_scene_node = Node(
    package="panda_moveit",
    executable="planning_scene_setup",
    name="planning_scene_setup",
    output="screen",
    parameters=[{"use_sim_time": True}],
    )

    delayed_planning_scene = TimerAction(
        period=8.0,
        actions=[planning_scene_node],
    )

    # ---------------------------------------------------------
    # Panda 1 gripper controller
    # ---------------------------------------------------------
    first_gripper_spawner = Node(
        package="controller_manager",
        executable="spawner",
        name="first_gripper_spawner",
        arguments=[
            "panda_gripper_controller",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "60",
        ],
        output="screen",
    )

    # ---------------------------------------------------------
    # Panda 2 gripper controller
    # ---------------------------------------------------------
    second_gripper_spawner = Node(
        package="controller_manager",
        executable="spawner",
        name="second_gripper_spawner",
        arguments=[
            "panda_gripper_controller",
            "--controller-manager",
            "/second/controller_manager",
            "--controller-manager-timeout",
            "60",
        ],
        output="screen",
    )

    return LaunchDescription([
        gazebo,
        dual_joint_states,
        dual_trajectory_adapter_node,

        robot_state_publisher,
        second_robot_state_publisher,

        clock_bridge,
        world_to_base_tf,
        world_to_second_tf,

        start_controllers,
        start_arm_controller,
        start_moveit,
        start_second_controllers,

        first_gripper_spawner,
        second_gripper_spawner,

        TimerAction(
            period=3.0,
            actions=[
                spawn_panda,
                spawn_second_panda,
            ],
        ),
        delayed_planning_scene,
    ])
