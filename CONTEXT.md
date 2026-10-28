# Project context

This document is for anyone who will change or extend this package: collaborators, students, or an AI assistant you hand the code to. It explains how the system fits together and why things are done the way they are. Most of the non-obvious choices came from a specific bug, and those are recorded here so they aren't undone by accident. For installation and usage, see [README.md](README.md).

## 1. Goal

**Research question:** two Franka Panda arms share a table. Given a pick-and-place task (move a cube into a drop-off tray, with obstacles in the way), which arm should do it?

- **No-model baseline:** plan the full task with both arms in MoveIt, then pick the cheaper plan. This is accurate but slow, about 0.8–1.1 s per decision in our runs.
- **Goal:** train a model that predicts each arm's cost directly. First a state-based MLP (scene description to cost), later a CNN on the overhead RGB-D image.
- **Comparison:** assignment quality and decision time against the planner baseline, nearest-arm and random choice.

**Cost of one arm doing one task:**

```
C = T_total + λ · L_total        (λ = 1.0 s/rad)
```

- `T_total` is the summed trajectory time in seconds, and `L_total` is the summed joint-space path length in radians, both over every motion stage: pregrasp, descend, lift, transport, lower and retreat.
- Infeasible tasks are flagged as infeasible and carry no cost. A fast failed attempt is never treated as a cheap one.
- Both terms are stored separately, so λ can be changed later.

Course context: Waterloo ECE 780, Robot Learning.

## 2. Status (2026-10-28)

**Working and tested:**
- Both arms pick and place reliably. Cubes land within about 0.5–6 mm of the target, and other cubes are left undisturbed.
- **Scenario randomizer:** trays, 3–6 obstacles, 12 cubes and the arm start poses. Re-applying a saved scenario restores an identical scene, including the arm joints.
- **`demonstrations.py`:** plans both arms, executes the cheaper one, and logs everything with before/after RGB-D. A 10-scenario dry run completed 10 of 10.
- **`collect.py`:** plan-only paired trials for large datasets, with every plan's joint trajectories (the training data for the generative policy).

**Not done yet:**
- A large dataset.
- Training: the MLP, then the CNN, plus the baselines.
- An executed validation subset showing that plan-only labels match execution. The early checks agree: in the 10-scenario dry run, 9 executed costs came within 4% of the planned cost, with one outlier at 27% (execution plans again, and OMPL's paths vary).

## 3. System architecture

```
Gazebo Harmonic (worlds/panda_world.sdf)
 ├─ model "panda"         (Panda 1, base at world x=0.2, y=0, z=1.025)
 │    gz_ros2_control → /controller_manager
 │       panda_arm_controller      (JointTrajectoryController, position)
 │       panda_gripper_controller  (JointGroupPositionController, 2 fingers)
 │       joint_state_broadcaster   → /joint_states
 ├─ model "panda_second"  (Panda 2, base at x=1.5, yawed by π)
 │    same, under the /second namespace → /second/joint_states
 ├─ 12 cubes, 2 trays, obstacles, overhead RGB-D camera
 └─ DetachableJoint systems, added at runtime when a cube is grasped

dual_joint_states.py      /joint_states + /second/joint_states → /dual_joint_states
                          (names prefixed first_panda_* / second_panda_*)
move_group (MoveIt 2)     one robot model "dual_panda" (config/urdf/dual_panda.urdf)
                          reads /dual_joint_states, plans with OMPL (RRTConnect)
dual_trajectory_adapter   MoveIt's /first|second/follow_joint_trajectory
                          → each arm's real Gazebo controller (strips the prefixes)
planning_scene_setup.py   loads table, cubes, trays and obstacles from the world into MoveIt
robot_state_publisher ×2  TF with frame_prefix first_ / second_, matching MoveIt's names
```

**One MoveIt model, two arms.** MoveIt sees a single robot with groups `first_panda_arm`, `second_panda_arm` and `both_panda_arms`. Each arm is planned on its own, and the other arm is a collision object at its *current* pose. No link pairs between the two arms have collision checking disabled in the SRDF. Planning for one arm assumes the other stays still, so **never move both arms at once**; the scripts always move one at a time.

**Frame names.** TF uses `first_panda_link8` and `second_panda_link8`, the same names as MoveIt. This works through `frame_prefix` on the two `robot_state_publisher` nodes, plus static transforms from `world` to each `*_panda_link0`. Gazebo and the controllers use the unprefixed names (`panda_joint1`, ...).

## 4. The pick-and-place pipeline (`pick_place.py`)

Each run goes through these stages in order:

1. `open_gripper` (execute mode only)
2. `grasp_select`
3. `pregrasp`: OMPL move to 15 cm above the cube
4. `descend`: Cartesian move to the cube
5. `close_gripper`, then `gz_attach` (execute mode only)
6. `lift`: Cartesian move up 10 cm
7. `transport`: OMPL move to above the tray
8. `lower`: Cartesian move into the tray
9. `gz_detach`, then `release` (execute mode only)
10. `retreat`: Cartesian move up 10 cm
11. `verify_place` (execute mode only)

**Plan-only mode** chains the stages. Each stage plans from the previous stage's end state, with the gripper width set to what it would be at that point. The cube is attached in MoveIt temporarily, and the planning scene is restored afterwards. Nothing moves in Gazebo.

### Key numbers and conventions

- **TCP offset:** 0.1034 m along link8's +Z, which is the centre of the finger pads and Franka's flange-to-TCP value. The fingertips are at 0.1122 m. Goals are given for the TCP and converted to link8: `T_world_link8 = T_world_tcp · inv(T_link8_tcp)`.
- **Grasp orientation:** link8 points straight down. For Panda 1 that's RPY (π, 0, −π/4), quaternion xyzw (0.924, −0.383, 0, 0); Panda 2's grasp is the same rotated by π. In this orientation the hand frame is aligned with the world, and the fingers close along world Y.
- **Cube rotation:** `grasp_select` tries the equivalent hand yaw closest to zero first, then the one turned 90°. The hand body is 0.21 m wide across the fingers but only 0.06 m along them, so the turned grasp often fits next to an obstacle where the first doesn't. Each yaw is first checked with collision-free IK (`/compute_ik`), which takes milliseconds. Without that check, an unreachable goal makes OMPL spend its full 8 s.
- **Placing:** from `transport` onward the hand returns to the standard orientation, so cubes are always placed square to the world.
- **Gripper:**
  - Each finger is commanded independently (`Float64MultiArray`, two values). Open is 0.04 per finger; closing commands 0.015, a little less than the cube's half-width, so the fingers squeeze.
  - A grasp counts only if the *gap* (finger1 + finger2) stalls above 0.036 m. Checking the gap rather than each finger matters, because the cube is rarely centred (for example 0.0237 / 0.0165).
- **Holding the cube:**
  - MoveIt's `AttachedCollisionObject` only affects planning; it doesn't hold anything in Gazebo. Gazebo uses a `DetachableJoint` between `panda_link7` (the hand is merged into link7 in the generated SDF) and the cube's `cube_link`.
  - The joint is attached only after the gripper check confirms a cube is held, and detached before the gripper opens.
- **Speed:** the default is 0.5 of the joint velocity and acceleration limits. The acceleration limits come from the Franka datasheet (15, 7.5, 10, 12.5, 15, 20, 20 rad/s²) and are set in the launch file. Every record stores `speed` and `accel`; keep them fixed within one dataset.

### Pitfalls that cost us time; keep these fixes

1. **Controller goal tolerances** (`config/ros2_controllers*.yaml`, `constraints`). Without them a trajectory controller reports success when the trajectory's time is up, even if the arm hasn't arrived. At higher speeds the next stage then started while the arm was still moving. That failed with `CONTROL_FAILED`, because MoveIt checks that the arm is within 0.01 rad of a trajectory's start. The tolerances make a trajectory succeed only once the arm is within 0.005 rad of the target and stopped, waiting up to 2 s for it.
2. **Wait for discovery before the first message.** A ROS or Gazebo message published, or a service called, before the other side has been discovered is *silently dropped*. This bit us with the first gripper command after launch and the first `/get_planning_scene` call. All clients now wait for their service, and publishers wait until they have a subscriber.
3. **DetachableJoint attaches as soon as it loads.** That's why it's added at runtime on the first grasp, through `/world/panda_world/entity/system/add`, rather than built into the robot model. Built in, it would lock every cube to the hand at startup. The model must be identified by **numeric entity id**: a lookup by name returns `true` but silently does nothing.
4. **Import order of `gz.msgs`.** With the protobuf version that ships with Jazzy, importing `entity_factory_pb2` before `pose_v_pb2` makes received `Pose_V` messages undecodable ("No message class registered for 'gz.msgs.Pose'"). All `gz.msgs` imports live at the top of `pick_place.py` with `Pose_V` first, and the other scripts import them from there. Also read Gazebo messages *inside* the callback; they can't be decoded once the subscription is gone.
5. **Don't use the `gz` command line from scripts.** It's slow (about 1 s per call) and occasionally hangs. Everything uses the `gz.transport13` Python bindings instead.
6. **Plan-only runs must restore the planning scene.** The `finally` block in `run()` does this. Attaching an object in MoveIt also removes the world object with the same id, so don't remove it separately; that leaves a half-applied scene.
7. **`finger_joint1` lower limit is −0.001** in `panda.urdf.xacro`. Before it was lowered from 0.0, one finger stayed at zero in Gazebo, and why the change helps was never pinned down. Leave it, and note that `dual_panda.urdf` still says 0.0 (MoveIt only).
8. **No URDF mimic for the fingers in Gazebo** (unsupported by Gazebo's physics engine). Both fingers are position-controlled independently. In `dual_panda.urdf`, finger2 mimics finger1, which MoveIt handles.

## 5. Scenario randomizer (`scenario.py`)

**Generation** (pure Python, deterministic in the seed):

| Element | Rules |
|---|---|
| Trays (2) | 0.18 m outer, 0.16 m inner, 3 cm walls. Placed at 0.35–0.78 m from *both* arm bases, at least 0.25 m apart, random yaw. The inner width lets open fingers, which reach 0.066 m either side of the cube, clear the walls. |
| Obstacles (3–6) | Walls: 15–35 cm long, 2–4 cm thick, 10–30 cm tall. Blocks: 6–15 cm wide, 6–20 cm tall. Random yaw. At least 0.25 m from base axes, and at least 0.12 m from other obstacles and trays. |
| Cubes (12) | 4 cm, random yaw in ±45°. The first 6 are reachable by both arms (0.30–0.76 m from both bases); the rest by at least one (0.30–0.72 m). At least 0.12 m apart, 0.08 m from obstacles, 0.06 m from trays. |
| Arm starts | `ready` plus uniform noise per joint. Accepted only if `/check_state_validity` reports both arms together as collision-free. They're chosen on the first apply and stored in the scenario file. |

**Applying a scenario,** in this order:
1. Release every cube, in Gazebo and MoveIt.
2. Move both arms to `ready`, so nothing can spawn inside an arm.
3. Remove the old `obstacle_*` and `dropoff_*` models.
4. Teleport the cubes, *before* spawning, so no cube lands inside a new obstacle.
5. Spawn the new obstacles and trays.
6. Sync MoveIt's planning scene.
7. Move the arms to their start poses.
8. Check that no cube drifted more than 1 cm.

It takes about 6–8 s.

## 6. Data

- **`demonstrations.py`:** one task per scenario. It plans both arms (best of 3), executes the cheaper one, and captures before/after RGB-D. The full field list is in the README.
- **`collect.py`:** plan-only, several tasks per scenario. `trials.jsonl` has one summary line per paired trial (task poses, both arms' feasibility and best cost); `trials/seed_<seed>_task_<i>.json` has every plan with stage-by-stage joint trajectories. It can resume by seed.
- **Best plan:** each arm's `best_plan_index` points to its cheapest feasible plan by full cost `C`, in both scripts. That plan is the arm's cost label.
- **Settings:** every trial stores `planner` (the MoveIt/OMPL settings from `PLANNER_SETTINGS` in `pick_place.py`), `cost_definition`, `speed` and `accel`.
- **Depth** is saved as compressed `.npz` (`np.load(f)['depth']`, float32 metres): lossless and about 11× smaller than `.npy`.
- **Splits:** train/validation/test must be split by seed, since a seed's tasks share one scene.

**Camera:**
- Fixed, looking straight down from (0.85, 0, 2.6); 960×720; 90° horizontal field of view; fx = fy = 480, cx = 480, cy = 360; no distortion.
- In the image, +x (towards Panda 2) is to the right and −y is down.
- `camera.json` stores the camera link pose and the optical-frame pose: z forward, x right, y down.
- The calibration was verified by projecting all 12 cubes into a captured frame. Each landed on a pixel of its own colour, with depth correct to within 1 mm.
- Depth is float32 metres, measured along the optical axis.

**Planning time:** `planning_time` per stage comes from MoveIt for OMPL stages, and is the measured service call time for Cartesian stages and for `grasp_select`. MoveIt reports 0 planning time for *executed* OMPL stages (planning and execution in one request), so use the plan-only attempts for the decision-time benchmark: `planner_assignment.baseline_decision_time`.

**Planner noise:** RRTConnect paths are random. Two plans of the same task differed by up to 1.5 rad, which is why each arm is planned several times and the best is kept.

## 7. Observations and open questions

- **Assignment balance (bias check, 40 seeds, 160 tasks, plan-only):** overall assignment 73 Panda 1 vs 82 Panda 2, and feasibility 75% vs 69%, so balanced. The 8-of-10 split in the first dry run was chance. When both arms were feasible (76 tasks), Panda 2 was cheaper in 47 (62%, median 0.57 lower). That's borderline significant (p ≈ 0.04) and worth rechecking on the full dataset. The generator itself is symmetric (checked over 2,000 seeds).
- **Distance barely predicts cost:** the arm nearer the cube and tray was the cheaper one in only 34 of those 76 tasks, mostly because the random start poses matter. A nearest-arm baseline will be weak.
- **Infeasibility is mostly reach.** Six cubes per scenario are forced into the region both arms can reach, so assignments there depend on obstacles and path cost.
- **Arms may clip at the image edge:** both arms are fully in frame at `ready`, but there's only about 30 px of margin.

## 8. Suggested next steps

1. Collect a plan-only dataset with `collect.py`: 2,000 seeds (about 8–11 h), split by seed into train, validation and test.
2. Collect an executed subset with `demonstrations.py` or `collect.py --execute`, and compare planned against executed costs.
3. **Features for the state MLP:** for each arm, the arm's start joints, the cube's pose relative to that arm's base, the tray's pose, and an obstacle summary (padded list or occupancy grid). Targets: feasibility (classification) and cost (regression).
4. **Baselines:** nearest arm (distance from base to cube plus cube to tray), random, and planner cost with its measured decision time.
5. RGB-D CNN on `before_rgb` / `before_depth`.

## 9. Regenerating generated files

- **`config/urdf/dual_panda.urdf`:** run `python3 create_dual_panda.py` from the package root. Check the finger limits and any manual edits afterwards, because it overwrites the file. Mesh paths are written as `package://`.
- **World:** edit `worlds/panda_world.sdf` directly. `planning_scene_setup.py` expects exactly 12 cubes (`EXPECTED_CUBES`). Cube and tray names are read from the world file by `pick_place.py`.
