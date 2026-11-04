# Guided diffusion sampling: test seeds 1850–1999

Date: 2026-11-04. This is the same checkpoint, execution and checks as [diffusion_fixed.md](diffusion_fixed.md) (`--exact-check --retime`). The only change is how the samples are drawn: the denoising process is steered toward collision-free trajectories that grasp and place accurately. This is the project's novel contribution.

## Method

A diffusion model builds a trajectory by starting from noise and removing it in 10 DDIM steps. At each step the network predicts the clean trajectory, x0. Guided sampling corrects that prediction before the next step. This happens at each of the **last 5 steps**:

1. Unnormalize the predicted x0 into joint angles.
2. Compute a cost on it with differentiable forward kinematics in PyTorch (`guidance.py`):
   - **Obstacle:** the arm is the same set of spheres as the fast collision check. Each sphere closer than 2 cm to a box, the table or the other arm adds (2 cm − gap)².
   - **Goal:** the squared distance from the gripper to the cube at the grasp frame, and to the place target at the release frame. The grasp and release frames are found from the gripper channel.
   - **Smoothness:** penalizes jerky corrections. It acts on the change made by guidance, not on the trajectory, so the model's own motion is untouched.
3. Take one gradient step on the joint angles, normalize again, and turn the corrected x0 back into the noise estimate that the DDIM step uses.

The gradient never goes through the U-Net, only through the forward kinematics, so a step costs about 8.5 ms for 16 samples × 2 arms. The early, noisy steps are not guided: there x0 is still a rough guess, and correcting it made the goal error worse.

**Settings:**
- Weights: obstacle 1, goal 0.4, smoothness 0.1.
- Margin 2 cm, 1 gradient step of size 1.0 at each of the last 5 steps.
- All other settings as in the earlier runs: 16 samples per arm, 10 DDIM steps, the 1 cm clearance rule for valid samples.

They were chosen on the validation seeds (1700–1849), not on the test seeds.

```bash
~/lerobot_venv/bin/python panda_moveit/evaluate_policy.py --checkpoint <ckpt> --seeds 1850 2000 \
    --save-trajectories --guide-obstacle 1 --guide-goal 0.4 --guide-iters 1 \
    --out ~/eval_exec_test_guided.jsonl
python3 panda_moveit/execute_policy.py --plans ~/eval_exec_test_guided.jsonl --out ~/exec_test_guided.jsonl \
    --exact-check --retime
```

Raw results: `~/exec_test_guided.jsonl`. The plans are in `~/eval_exec_test_guided.jsonl`.

## Ablation on the validation seeds (600 tasks, offline, no Gazebo)

| Guidance | Valid samples | Collision-free samples | Grasp / place error | Missed though doable | Sampling time |
|---|---|---|---|---|---|
| None | 83.2% | 95.3% | 1.2 / 1.1 cm | 12 | 189 ms |
| Obstacle only | 85.0% | 99.6% | 1.2 / 1.1 cm | 3 | 313 ms |
| Goal only | 90.3% | 95.4% | 0.5 / 0.5 cm | 5 | 281 ms |
| **Both** | **92.5%** | **99.7%** | **0.6 / 0.5 cm** | **1** | 322 ms |

Each term does its own job, and the two combine. The obstacle term removes nearly all near-collisions, and the goal term halves the grasp and place error. Arm-choice agreement with MoveIt stayed at about 85% in all four settings, so guidance improves the trajectories, not the choice of arm.

## Results

| Metric | Guided | Fixed (no guidance) |
|---|---|---|
| Tasks | 600 (560 executed, 40 with no arm) | 551 executed |
| **Success, doable tasks** | **535/583 (91.8%)** | 85.9% |
| Success, all tasks | 539/600 (89.8%) | 83.8% |
| Success, executed tasks | 539/560 (96.2%) | 91.3% |
| Chose the best arm (labels) | 502/600 (83.7%) | 83.2% |
| Chose an arm that can't do the task | 8/600 (1.3%) | 1.0% |
| No arm though the labels have one | 27 | 34 |
| Correctly said "neither" | 13/17 | 15/17 |
| **Collision signs** (executed tasks) | **5/560 (0.9%)** | 0.5% |
| MoveIt contact along the motion | 0/560 | 0/551 |
| Arm held back in Gazebo | 5/560 (0.9%) | 0.5% |
| Other cubes knocked | 0 | 0 |

Times in seconds, given as median | mean | p90.

| Metric | Median | Mean | p90 | Fixed median (p90) |
|---|---|---|---|---|
| **Time to plan** (sampling + scoring + exact check) | **0.40** | 0.40 | 0.45 | 0.27 (0.30) |
| Exact check alone, ms | 3.7 | 5.9 | 11.8 | 5.5 (11.3) |
| Motion time (successful tasks) | 4.30 | 4.47 | 5.38 | 4.35 (5.60) |
| Execution wall time (successful tasks) | 6.90 | 7.13 | 8.48 | 6.98 (8.61) |
| **Plan + motion** (successful tasks) | **4.69** | 4.87 | 5.82 | 4.62 (5.89) |
| Cost T + L, executed (successful tasks) | 8.55 | 8.97 | 11.26 | 8.80 (11.45) |
| Place error, cm (executed tasks) | 0.48 | 1.05 | 1.06 | 0.65 (1.39) |

The time to plan breaks down into sampling at 332 ms (189 ms without guidance), the fast checks at 67 ms and the exact check at 4 ms.

**Failures (61, against 97):**
- **15 at the place stage**, down from 46.
- **36 with no clear sample**, up from 30. 20 of these also had none in the fixed run. Of the other 16, the fixed run chose no arm for 7: guidance made a sample pass the fast check, but the exact check still found a contact. The fixed run succeeded on 6 and failed on 3. Sampling is random, so some tasks fall either side of the line between runs.
- **4 where the decision rule said "neither"**, down from 19.
- **4 missed grasps** (up from 2) and **1 gripper that failed to close.**
- **1 Gazebo error:** a timeout waiting for the world pose message (seed 1901, task 3). The arm never moved, so this is infrastructure, not the policy.

Task by task, guidance turned 50 failures into successes and 14 successes into failures.

## What this showed

- **Success on doable tasks rose from 85.9% to 91.8%,** closing about half the gap to MoveIt (97.8%). Most of the gain is placement: place failures fell from 46 to 15, and the median place error from 0.65 to 0.48 cm.
- **Speed is unchanged where it matters.** Planning takes 0.13 s longer, but plan + motion is still 4.69 s against MoveIt's 5.22 s. Cost is now lower than MoveIt's: 8.55 against 8.91.
- **Collisions stay near zero:** MoveIt's checker found no contacts. The 5 collision signs were the arm held back in Gazebo, and only 1 of them failed.
- **Arm choice did not improve** (83.7% against MoveIt's 95.0%). Guidance makes each sample better but doesn't change which arm the samples favour. That is what the next step, a learned arm classifier on the sample statistics, is meant to address.
- **The remaining failures are mostly tasks with no collision-free sample.** Guidance only adjusts the last 5 steps, so it can't move a sample that goes the wrong way around an obstacle.
