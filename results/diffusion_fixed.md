# Diffusion policy with exact collision check and retiming: test seeds 1850–1999

Date: 2026-11-03. This is the same policy and checkpoint as [diffusion_original.md](diffusion_original.md), with two fixes in how the executed trajectory is chosen and timed. It uses fresh samples.

## What changed

1. **Exact collision check when choosing (`--exact-check`).**
   - The samples are tried in order: the chosen arm's samples within 1 cm of the cube and the target, cheapest first; then its other samples, most accurate first; then the other arm's.
   - The first sample with no contact in MoveIt's exact collision model is executed. If none is clear, the policy declines the task.
   - The check runs MoveIt's collision model in-process on a copy of the scene. It takes about 5 ms per trajectory, against 2–6 s through the MoveIt service, and its time is added to the time to plan.
   - It was validated against the MoveIt service on 59 trajectories (9,323 states) and agreed on all of them.
2. **Retiming (`--retime`).**
   - Frames are skipped until a joint has moved 0.01 rad, which removes the near-still idle time after the release.
   - Where the arm moves, the sampled 5 Hz timing is kept. No joint exceeds the joint velocity limit scaled by 0.5.

Everything else is the same as the original run: sampling, scoring, decision rule, execution and success test.

```bash
~/lerobot_venv/bin/python panda_moveit/evaluate_policy.py --checkpoint <ckpt> --seeds 1850 2000 \
    --save-trajectories --out ~/eval_exec_test_v2.jsonl
python3 panda_moveit/execute_policy.py --plans ~/eval_exec_test_v2.jsonl --out ~/exec_test_fixed.jsonl \
    --exact-check --retime
```

Raw results: `~/exec_test_fixed.jsonl`. The plans are in `~/eval_exec_test_v2.jsonl`; with the new samples the arm choice agrees with MoveIt 84.7% of the time. An earlier attempt had a bug in the local scene copy (obstacles from earlier tasks accumulated) and is kept, not used, as `~/exec_test_fixed_bad.jsonl`.

## Results

| Metric | Value | Original |
|---|---|---|
| Tasks | 600 (551 executed, 49 with no arm) | 577 executed |
| **Success, doable tasks** | **501/583 (85.9%)** | 86.1% |
| Success, all tasks | 503/600 (83.8%) | 84.2% |
| Success, executed tasks | 503/551 (91.3%) | 87.5% |
| Chose the best arm (labels) | 499/600 (83.2%) | 83.5% |
| Chose an arm that can't do the task | 6/600 (1.0%) | 3.0% |
| No arm though the labels have one | 34 | 15 |
| Correctly said "neither" | **15/17** | 8/17 |
| **Collision signs** (executed tasks) | **3/551 (0.5%)** | 11.8% |
| MoveIt contact along the motion | 0/551 | 11.3% |
| Arm held back in Gazebo | 3/551 (0.5%) | 5.0% |
| Other cubes knocked | 0 | 0 |

Times in seconds, given as median | mean | p90.

| Metric | Median | Mean | p90 | Original median (p90) |
|---|---|---|---|---|
| **Time to plan** (sampling + scoring + exact check) | **0.27** | 0.27 | 0.30 | 0.26 (0.29) |
| Exact check alone, ms | 5.5 | 6.7 | 11.3 | — |
| Motion time (successful tasks) | 4.35 | 4.55 | 5.60 | 5.80 (12.80) |
| Same tasks at the sampled timing | 5.80 | 7.14 | 12.60 | — |
| Execution wall time (successful tasks) | 6.98 | 7.25 | 8.61 | 8.30 (15.35) |
| **Plan + motion** (successful tasks) | **4.62** | 4.82 | 5.89 | 6.05 (13.03) |
| Cost T + L, executed (successful tasks) | 8.80 | 9.13 | 11.45 | 10.31 (17.86) |
| Place error, cm (executed tasks) | 0.65 | 1.01 | 1.39 | 0.66 (1.80) |

**The exact check:**
- It rejected the first-choice sample in 70 of 581 tasks; without the check, these would have been executed and would have collided.
- It found a clear alternative in 40 of those.
- In 30 tasks none of the 32 samples was clear, so the policy declined.

**Failures (97):**
- **46 at the place stage**, with no collision: the cube ended up more than 1.5 cm from the target.
- **30 with no clear sample.**
- **19 where the decision rule said "neither".**
- **2 missed grasps.**

## What this showed

- **Collisions fell from 11.8% to 0.5%.** The 3 remaining were the arm held back in Gazebo; MoveIt's checker found no contacts.
- **Diffusion now beats MoveIt end to end:** plan + motion takes 4.62 s against 5.22 s. Its slow tail matches MoveIt's, at a p90 of 5.89 s against 6.46 s. Its cost (8.80) is level with MoveIt's (8.91).
- **Success on doable tasks didn't change** (85.9%). Collision failures turned into refusals, so the policy now fails safely, by declining, instead of hitting things.
- **The remaining gap to MoveIt (97.8%) is about sample quality:** placement accuracy, and tasks with no clear sample. Guided sampling targets these: see [diffusion_guided.md](diffusion_guided.md).
