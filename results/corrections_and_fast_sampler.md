# Corrections, MoveIt fallback and the faster sampler: test seeds 1850–1999

Date: 2026-11-15. An audit of the code found two measurement bugs that affected the earlier results. This file records the fixes, the numbers they changed, and two additions made afterwards: a MoveIt fallback for declined tasks and a faster sampler. The numbers here replace those in the earlier files wherever they differ.

## 1. MoveIt's time to plan was overstated

`pick_place.py` waited for each MoveIt reply by sleeping 30 ms at a time, so every planning call was rounded up and the time to plan had a floor at 0.79 s. The wait now returns as soon as the reply arrives. MoveIt's planning was re-timed on all 600 test tasks, plan-only, on an idle PC.

| MoveIt time to plan (both arms) | Median | Mean | p90 |
|---|---|---|---|
| Earlier measurement (with the polling delay) | 0.88 s | 0.92 s | 1.07 s |
| **Re-measured, arms one after the other** | **0.38 s** | 0.39 s | 0.42 s |
| Re-measured, reusing the grasp-selection plans (`--reuse-grasp-plans`) | 0.32 s | 0.31 s | 0.36 s |
| Arms planned in parallel (estimate: the slower arm's time per task) | 0.20 s | 0.21 s | 0.22 s |

- Plan + motion for MoveIt, with the re-measured planning time and the motion times of the original run: 4.64 s (p90 6.04) sequential, 4.45 s (p90 5.85) parallel.
- The parallel figure is an estimate and slightly favours MoveIt: two planners running at once would share the CPU.
- Execution results for MoveIt (success, cost, arm choice) are unchanged; only the planning time was affected.
- Two independent single MoveIt runs on the test tasks chose the same arm in 93.3% of tasks, and 85 of the 265 tasks that both arms can do are near-ties (costs within 0.5). Agreement with MoveIt's labels therefore has a ceiling well below 100%.

```bash
python3 panda_moveit/execute_moveit.py --seeds 1850 2000 --plan-only --out ~/moveit_timing/test.jsonl
python3 panda_moveit/execute_moveit.py --seeds 1850 2000 --plan-only --reuse-grasp-plans --out ~/moveit_timing/test_reuse.jsonl
```

## 2. The exact check wrongly rejected the other arm's samples

The in-process collision checker kept one robot state. When a candidate of one arm collided, that arm was left where it collided, so every candidate of the other arm was then checked against an arm frozen mid-motion and usually failed. The checker now writes both arms for every state: the checked arm at the candidate's state and the other arm at its start pose.

Only trials where the first candidate had been rejected could change. Those 58 trials per run were executed again; the rest keep their rows. Raw: `~/exec_fix/`.

| Success, doable tasks | Before the fix | After |
|---|---|---|
| Diffusion | 85.9% | 86.3% |
| Diffusion + NN-A | 86.3% | 86.4% |
| Diffusion + NN-B | 85.8% | 86.1% |
| Diffusion + Guidance | 91.8% | 92.5% |
| Guidance + NN-A | 92.1% | 92.5% |
| Guidance + NN-B | 91.9% | 92.5% |
| Guidance + NN-A v2 | 91.9% | 92.5% |
| Guidance + NN-A + resampling | 92.6% | 92.8% |

The effect is small (at most +0.7 points). With guidance, the doable tasks declined for lack of an arm fell from 27 to 22.

## 3. MoveIt fallback for declined tasks

When the policy declines a task (no arm has a usable sample, or no sample passes the exact check), `execute_policy.py --moveit-fallback` hands that task to MoveIt: MoveIt plans both arms and executes the cheaper plan, or reports that neither arm can do it. MoveIt's planning time is added to the task's time to plan.

With the earlier sampler (Guidance + NN-A): 35 declined trials, MoveIt completed 20 and found no plan for the other 15.

| Metric | Guidance + NN-A | + MoveIt fallback | MoveIt |
|---|---|---|---|
| Success, doable tasks | 92.5% | **95.7%** | 97.8% |
| Chose the best arm | 86.0% | 89.0% | 95.0% |
| No arm though one could | 22 | 3 | 7 |
| Time to plan, median (mean) | 0.40 s (0.40) | 0.41 s (0.43) | 0.38 s (0.39) |

## 4. Faster sampler

With MoveIt correctly timed at 0.38 s, guided diffusion at 0.40 s was no longer faster. Profiling showed where the time went: the U-Net takes 8.3 ms per denoising step, but each task sampled two 16-step chunks and the fast collision check ran on the CPU.

1. **Stop when the motion is finished.** A sample counts as finished once it has released the cube and retreated for 5 frames (or stopped moving). The second chunk is drawn only if no arm has a usable sample after the first. Before, a second chunk was drawn in most tasks.
2. **Fast collision check on the GPU** (`Guide.clearance`), using the same sphere model as guidance: 64 ms → about 10 ms.
3. **Palm spheres corrected.** The hand's spheres were laid across the wrong axis (off by 45°) and were too narrow. With the fix, the fast check agrees better with the exact check: on the validation seeds the exact check rejected the first candidate in 54 tasks instead of 79, and the doable tasks declined fell from 16 to 11.

The diffusion model was not retrained. The arm-choice networks were retrained on samples from the new sampler, guided and unguided separately (same recipe as v2: soft targets with τ = 0.25, 5 networks, trained on the teammate's 3,996 tasks plus the 600 validation tasks for the cross-validation median number of epochs: 35 to 51).

Validation seeds (600 tasks, offline, idle PC):

| | Sampling + fast checks | Valid samples | Agrees with MoveIt | Missed though doable |
|---|---|---|---|---|
| Guided, before | 0.37 s | 92.5% | 84.7% | 2 |
| **Guided, faster sampler** | **0.16 s** | 90.0% | 86.5% | 3 |
| Unguided, faster sampler | 0.09 s | 81.5% | 85.8% | 18 |

```bash
~/lerobot_venv/bin/python panda_moveit/evaluate_policy.py --checkpoint <ckpt> --seeds 1850 2000 --save-trajectories \
    --guide-obstacle 1 --guide-goal 0.4 --guide-iters 1 --out ~/eval_fast/test_guided.jsonl
python3 panda_moveit/execute_policy.py --plans ~/eval_fast/test_guided.jsonl --out ~/eval_fast/exec_guided.jsonl \
    --exact-check --retime [--moveit-fallback]
```

The whole sequence is in `~/eval_fast/run_test.sh`. Raw: `~/eval_fast/exec_{guided,nn,fallback,resample}.jsonl`.

## Results with the faster sampler (600 test tasks, Gazebo)

| | Diffusion | Diffusion + NN-A | Diffusion + NN-B | Guidance | Guidance + NN-A | Guidance + NN-B | Guidance + NN-A + resampling | Guidance + NN-A + MoveIt fallback | MoveIt | MoveIt, arms in parallel (estimate) |
|---|---|---|---|---|---|---|---|---|---|---|
| Time to plan, median | **0.10 s** | 0.10 s | 0.10 s | 0.16 s | 0.16 s | 0.16 s | 0.16 s | 0.16 s | 0.38 s | 0.20 s |
| Plan + motion, median (p90) | 4.41 s (5.35) | 4.41 s (5.33) | 4.41 s (5.33) | 4.45 s (5.44) | 4.45 s (5.45) | 4.45 s (5.45) | 4.46 s (5.45) | 4.46 s (5.51) | 4.64 s (6.04) | 4.45 s (5.85) |
| Cost T + L, median | 8.65 | 8.60 | 8.60 | 8.52 | 8.49 | 8.50 | 8.49 | 8.52 | 8.91 | 8.91 |
| Success, doable tasks | 86.6% | 86.8% | 86.8% | 93.8% | 93.8% | 94.0% | 94.5% | **96.2%** | 97.8% | 97.8% |
| Collision signs | 0.9% | 0.9% | 0.9% | 1.0% | 0.9% | 0.9% | 1.0% | 0.8% | planned collision-free | planned collision-free |
| Chose the best arm | 84.0% | 86.8% | 86.7% | 86.0% | 86.2% | 87.5% | 87.2% | 88.3% | 95.0% | 95.0% |
| No arm though one could | 29 | 29 | 29 | 18 | 18 | 18 | 10 | 3 | 7 | 7 |
| Correctly said "neither" | 16/17 | 16/17 | 16/17 | 10/17 | 10/17 | 10/17 | 8/17 | 8/17 | 17/17 | 17/17 |

Time to plan with guidance: sampling 148 ms + fast checks 9 ms + exact check 4 ms. Without guidance: 85 ms + 9 ms + 5 ms. For the NN columns only the trials where the network changed the rule's arm were executed (43 to 53 per column); the others are the rule run's rows.

Paired tests on the same tasks (exact McNemar):

| Comparison | Only the first succeeded (or chose the best arm) / only the second | p |
|---|---|---|
| Success: faster sampler vs earlier sampler (Guidance) | 23 / 14 | 0.19 |
| Success: MoveIt vs Guidance | 33 / 15 | 0.013 |
| Success: MoveIt vs Guidance + NN-A + MoveIt fallback | 21 / 18 | 0.75 |
| Success: Guidance vs Diffusion (no guidance) | 59 / 13 | < 0.001 |
| Best arm: Diffusion + NN-A vs rule | 35 / 18 | 0.027 |
| Best arm: Diffusion + NN-B vs rule | 33 / 17 | 0.033 |
| Best arm: Guidance + NN-A vs rule | 22 / 21 | 1.0 |
| Best arm: Guidance + NN-B vs rule | 26 / 17 | 0.22 |

- **Fallback:** 28 trials were declined. MoveIt completed 15 of them, failed during execution on 2, and found no plan for 11. Those 28 trials took a median 0.66 s to plan; the other 572 are unaffected.
- **Resampling:** of the same 28 trials, a second round found an arm for 10, and 4 of those succeeded.
- **Cost against MoveIt on the 537 tasks both completed:** the policy's executed cost is lower by a median 0.12 (mean 0.26), and lower in 59% of tasks.
- **Speed against MoveIt task by task:** faster than sequential MoveIt in 99.8% of tasks, and faster than the parallel estimate in 98.0% (median 0.04 s faster).

## What this showed

- **The speed claim holds again, on a fair baseline.** Guided diffusion decides in 0.16 s against 0.38 s for MoveIt as it runs here (2.4 times faster) and against an estimated 0.20 s if MoveIt planned both arms in parallel. Plan + motion is 4.45 s against 4.64 s; against parallel MoveIt the medians are equal, and diffusion has the shorter tail (p90 5.44 s against 5.85 s).
- **The earlier "0.40 s against 0.88 s" comparison was wrong** on the MoveIt side and should not be quoted.
- **Motions are cheaper than MoveIt's:** median cost 8.52 against 8.91.
- **Reliability: 93.8% alone, 96.2% with the MoveIt fallback.** Alone, the policy still succeeds less often than MoveIt (p = 0.013). With the fallback, the difference from MoveIt (97.8%) is no longer statistically detectable on 600 tasks (p = 0.75), while the median time to plan stays at 0.16 s because only 28 of 600 tasks use it.
- **It makes fewer correct refusals.** The policy attempted 7 of the 17 tasks that no arm can do (it had attempted 4 with the earlier sampler), and these attempts fail.
- **Guidance is the main gain in reliability:** 86.6% → 93.8% success on doable tasks (p < 0.001), for 0.06 s more planning. Without guidance diffusion plans in 0.10 s.
- **The learned arm choice helps where the rule is weakest.** Without guidance, the network chose the best arm in 86.8% of tasks against the rule's 84.0% (p = 0.027; NN-B 86.7%, p = 0.033) and lowered the median cost from 8.65 to 8.60. With guidance the rule is already at 86.0% (the corrected palm model and guidance make its samples more informative), and the networks' gains (86.2% and 87.5%) are not statistically detectable on 600 tasks. In cross-validation on 3,996 tasks all four networks beat the rule by 2.4 to 3.6 points. Success doesn't change in any column: most failures happen on the right arm.
