# MoveIt baseline: test seeds 1850–1999

Date: 2026-11-03. No learning is involved here. This is the planner the diffusion policy is compared against.

## What was run

The 600 test tasks are 150 seeds × 4 tasks. Each task gives one arm a cube to pick and a place target. Every task was run in Gazebo as follows:

1. Load the task's scene into Gazebo and MoveIt: cubes, obstacles, trays and both arm start poses.
2. **Decision:** MoveIt plans the pick-and-place once with each arm, without executing. Both plans together are timed as the time to plan.
3. Pick the arm whose plan has the lower cost, or neither if neither arm has a plan.
4. **Execution:** that arm plans and executes the pick-and-place in Gazebo, stage by stage.
5. **Success:** the cube is within 1.5 cm of the place target 1 s after release.

- Speed and acceleration scaling: 0.5 of the joint limits.
- Cost = T + 1.0 · L, where T is the motion time in seconds and L is the joint path length in radians.
- Labels: from the plan-only dataset. For each arm, the best of 3 planning repeats; the cheaper feasible arm is the "best arm".

```bash
python3 panda_moveit/execute_moveit.py --seeds 1850 2000 --out ~/moveit_test.jsonl
```

Raw results: `~/moveit_test.jsonl`, one row per task.

## Results

| Metric | Value |
|---|---|
| Tasks | 600 (576 executed, 24 with no plan for either arm) |
| **Success, doable tasks** (labels have an arm) | **570/583 (97.8%)** |
| Success, all tasks | 570/600 (95.0%) |
| Success, executed tasks | 570/576 (99.0%) |
| Chose the best arm (labels) | 570/600 (95.0%) |
| Chose an arm that can't do the task | 0/600 |
| No plan though the labels have one | 7 |
| Correctly said "neither" | 17/17 |
| Other cubes knocked | 0 |

Times in seconds, given as median | mean | p90. p90 means 90% of tasks took less than this.

| Metric | Median | Mean | p90 |
|---|---|---|---|
| **Time to plan** (both arms, one plan each) | **0.88** | 0.92 | 1.07 |
| Motion time (successful tasks) | 4.25 | 4.48 | 5.64 |
| Execution wall time (successful tasks) | 8.26 | 8.52 | 9.75 |
| Re-planning during execution (successful tasks) | 0.27 | 0.28 | 0.30 |
| **Plan + motion** (successful tasks) | **5.22** | 5.42 | 6.46 |
| Cost T + L, executed (successful tasks) | 8.91 | 9.70 | 13.07 |
| Place error, cm (successful tasks) | 0.05 | 0.12 | 0.38 |

**Failures (30):**
- 24 tasks had no plan for either arm. For 17 of them the labels also have none, so "neither" was correct; for the other 7 the labels have a plan.
- 6 failed during execution: 2 in descend, 2 in transport, 1 in lower and 1 in grasp_select.

## Notes

- Wall time is longer than motion time because it also covers gripper actions, settling and per-stage planning during execution.
- MoveIt's motions are collision-free by construction. No collision signals were measured for this baseline, and no other cube was knocked.
- The 1.2% of doable tasks with no plan come from planning randomness: the labels keep the best of 3 attempts, and this run made only one.
