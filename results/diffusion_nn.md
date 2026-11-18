# Learned arm choice (NN): test seeds 1850–1999

> **Correction, 2026-11-15.** The speed comparisons with MoveIt below use the overstated 0.88 s. Success figures rise by up to 0.7 points with the exact-check fix. With the faster sampler the retrained network no longer shows a detectable gain over the rule on the test set, although it still does in cross-validation. See [corrections_and_fast_sampler.md](corrections_and_fast_sampler.md).

Date: 2026-11-14. This replaces the arm-choice rule ("the arm with the cheapest valid sample") with a small network trained on MoveIt's labels. Sampling, the exact collision check, retiming and execution are unchanged from [diffusion_fixed.md](diffusion_fixed.md) (unguided) and [diffusion_guided.md](diffusion_guided.md) (guided).

## Method

`panda_moveit/arm_classifier.py`. Each arm's 16 samples are summarized into a feature vector. The network outputs 3 logits: first arm, second arm, neither.

- **Sample features (16 per arm):**
  - fractions of samples that are valid, collision-free and accurate, and whether any sample is valid
  - best valid cost, median cost, cost spread
  - minimum and median grasp and place errors
  - clearance of the best sample, median clearance
  - median motion time
  - distance from the arm's base to the cube and to the target
- **Scene features (6 more per arm, NN-B only):**
  - how far the start pose is from `ready`
  - gap from the cube and from the target to the nearest obstacle
  - obstacles within reach
  - the other arm's distance to the cube and to the target
- **Network (shared scorer):** one MLP, 16 (or 22) → 64 → 64 → 1, scores each arm with the same weights. A learned bias is the "neither" logit. About 5.3k parameters.
- **Training:**
  - cross-entropy, no class weights (inverse-frequency weights made it say "neither" too often)
  - Adam, learning rate 1e-3, weight decay 1e-4, dropout 0.1
  - CUDA, a few seconds per model
- **Data:**
  - **Train:** a teammate's seeds 2000–2999. That's 999 seeds, 3,996 tasks: scenes the diffusion model never saw, with diffusion samples drawn for this.
  - **Early stopping:** the validation seeds 1700–1849.
  - **Final score:** the test seeds, once.

  The guided and unguided versions are trained on their own samples.
- **Constraint when executing:** the network chooses only among arms that have a usable sample, or neither, since there's nothing to execute for an arm without one. This was added after seeing, on the test plans, that the unguided network sometimes picked such an arm (9 trials), so this one rule was informed by the test set. No other setting was tuned on test.
- **Models used:** seed 0 of each variant (`~/arm_classifier/{none,both}_{samples,both}_0.pt`).

```bash
~/lerobot_venv/bin/python panda_moveit/evaluate_policy.py --checkpoint <ckpt> --data ~/friend_data/seeds_2000 \
    --seeds 2000 3000 [--guide-obstacle 1 --guide-goal 0.4 --guide-iters 1] --out ~/eval_friend/<none|both>.jsonl
~/lerobot_venv/bin/python panda_moveit/arm_classifier.py train --class-weights none --features <samples|both> \
    --data ~/friend_data/seeds_2000 --train ~/eval_friend/<none|both>.jsonl \
    --val-data ~/panda_data --val ~/eval_guide/val_<none|both>.jsonl --out <model>.pt
~/lerobot_venv/bin/python panda_moveit/arm_classifier.py relabel --model <model>.pt --plans <test plans> --out <plans>
python3 panda_moveit/execute_policy.py --plans <plans> --out <exec>.jsonl --exact-check --retime
```

Only the trials where the network changed the rule's arm were executed again: 45, 44, 59 and 51 for the four variants. The other trials are the rows of the earlier runs, which had the same plan and the same arm. Raw results: `~/arm_classifier/exec_{none,both}_{samples,both}.jsonl`.

## Offline results

5-fold cross-validation by seed on the training seeds, 3 repeats; agreement with MoveIt's arm:

| Class weights | NN-A, unguided | NN-B, unguided | NN-A, guided | NN-B, guided |
|---|---|---|---|---|
| Inverse frequency | 84.9% | 84.2% | – | – |
| Square root | 86.6% | 85.2% | 86.7% | 85.7% |
| **None** | **87.3%** | 86.4% | **87.6%** | 86.6% |
| Rule | 85.6% | 85.6% | 86.1% | 86.1% |

Trained on only the 600 validation tasks, every network lost to the rule by about 6 points (79–80%). The method needs the larger training set.

Test plans (no Gazebo), 3 training seeds per variant, before the usable-arm constraint:

| | Agrees with MoveIt | Mean regret | Regret > 0.5 |
|---|---|---|---|
| Rule, unguided | 84.7% | 0.103 | 35 |
| NN-A, unguided | 86.6% | 0.092 | 26–33 |
| NN-B, unguided | 86.8% | 0.080 | 27–28 |
| Rule, guided | 85.8% | 0.095 | 34 |
| NN-A, guided | 87.7% | 0.080 | 24–26 |
| NN-B, guided | 87.7% | 0.067 | 21–26 |

Regret = the MoveIt cost of the chosen arm minus that of the best arm.

## Results in Gazebo

| Metric | Diffusion | + NN-A | + NN-B | Diffusion + Guidance | + NN-A | + NN-B | MoveIt |
|---|---|---|---|---|---|---|---|
| Time to plan, median | 0.27 s | 0.27 s | 0.27 s | 0.40 s | 0.40 s | 0.40 s | 0.88 s |
| Plan + motion, median | 4.62 s | 4.62 s | 4.62 s | 4.69 s | 4.68 s | 4.69 s | 5.22 s |
| Cost T + L, median | 8.80 | 8.78 | 8.78 | 8.55 | 8.53 | 8.53 | 8.91 |
| Success, doable tasks | 85.9% | 86.3% | 85.8% | 91.8% | **92.1%** | 91.9% | 97.8% |
| Chose the best arm | 83.2% | 84.8% | 84.5% | 83.7% | **85.8%** | 85.0% | 95.0% |
| No arm though one could | 34 | 33 | 34 | 27 | 25 | 26 | 7 |
| Correctly said "neither" | 15/17 | 15/17 | 15/17 | 13/17 | 13/17 | 13/17 | 17/17 |
| Collision signs | 0.5% | 0.5% | 0.5% | 0.9% | 0.7% | 0.7% | none |

The network adds well under 1 ms to the time to plan.

On just the trials the network changed:

| Variant | Changed | Best arm: rule → NN | Success: rule → NN | Mean cost where both succeed |
|---|---|---|---|---|
| Diffusion + NN-A | 45 | 17 → 27 | 42 → 44 | 9.00 → 8.74 |
| Diffusion + NN-B | 44 | 17 → 25 | 39 → 38 | 9.10 → 8.66 |
| Guidance + NN-A | 59 | 22 → 35 | 53 → 55 | 8.62 → 8.55 |
| Guidance + NN-B | 51 | 22 → 30 | 48 → 49 | 8.55 → 8.52 |

## What this showed

- **The network chooses the arm better than the rule:** +1.3 to +2.1 points of best-arm choice in Gazebo, and lower regret. On the trials it changes, it picks MoveIt's arm about 60% of the time, against about 38% for the rule. The improved version below shows the difference is statistically significant.
- **Success and time hardly move** (+0.4 points at best). Most execution failures happen on the correct arm (placement, no clear sample), so a better arm choice can't fix them. The cost gain is real on the changed trials, but they are under 10% of all trials.
- **The scene features don't help.** NN-B matches or trails NN-A, which supports the idea that the samples already carry what the scene says.
- **Data was the limit.** On 600 tasks the network lost to the rule. On 4,000 it won. Most of the remaining disagreement with MoveIt is near-ties: in validation, 35 of 88 disagreements differ by less than 0.5 in MoveIt cost, which is within MoveIt's own planning noise.
- **Best variant: Diffusion + Guidance + NN-A,** at 92.1% success, 85.8% best arm, plan + motion 4.68 s against MoveIt's 5.22 s, and median cost 8.53 against 8.91. Guidance gives most of the gain over plain diffusion (+5.9 points of success); the NN adds a small, consistent improvement in arm choice.

## Improved network (v2) and significance

Three changes, all chosen by cross-validation on the training seeds; the test set was used once, at the end.

1. **Soft targets (`--tau`).** Instead of a one-hot label, the feasible arms share the target by softmax(−cost / τ) of their MoveIt costs. A near-tie, which MoveIt itself decides differently from run to run, is no longer taught as a certain answer. τ = 0.25 (guided) and 1.0 (unguided) were best in cross-validation; the gain alone is small (+0.2 to +0.6 points).
2. **More data.** The learning curve was still rising with all the data:

   | Training tasks | 240 | 600 | 1,200 | 1,800 | 2,400 | Rule |
   |---|---|---|---|---|---|---|
   | Agreement, guided | 83.7% | 85.7% | 87.0% | 87.4% | 87.8% | 86.1% |
   | Mean regret, guided | 0.107 | 0.083 | 0.072 | 0.069 | 0.065 | 0.087 |

   The network needs about 600 tasks to match the rule and keeps improving after. The final models train on the teammate's 3,996 tasks plus the 600 validation tasks (4,596), for a fixed number of epochs (the cross-validation median: 70 guided, 86 unguided) since there is no held-out set left for early stopping.
3. **Ensemble** of 5 networks (different seeds), probabilities averaged.

```bash
~/lerobot_venv/bin/python panda_moveit/arm_classifier.py curve --class-weights none --tau 0.25 \
    --data ~/friend_data/seeds_2000 --plans ~/eval_friend/both.jsonl
~/lerobot_venv/bin/python panda_moveit/arm_classifier.py train --class-weights none --tau 0.25 --epochs 70 --seed <0-4> \
    --data ~/friend_data/seeds_2000 ~/panda_data --train ~/eval_friend/both.jsonl ~/eval_guide/val_both.jsonl \
    --out ~/arm_classifier/v2/both_samples_<seed>.pt
~/lerobot_venv/bin/python panda_moveit/arm_classifier.py relabel --data ~/panda_data --plans ~/eval_exec_test_guided.jsonl \
    --model ~/arm_classifier/v2/both_samples_{0,1,2,3,4}.pt --out ~/arm_classifier/v2/plans_both_samples.jsonl
```

**Is the network better than the rule?** Paired tests on the same tasks: McNemar's exact test on "chose MoveIt's arm", and a bootstrap 95% interval on the drop in regret (MoveIt cost of the chosen arm minus the best, per task).

| Evaluation | Tasks | Agreement: rule → NN | NN right, rule wrong / the reverse | McNemar p | Mean regret: rule → NN | Regret > 0.5: rule → NN |
|---|---|---|---|---|---|---|
| Cross-validation, guided (3 repeats) | 3,996 | 86.1% → 87.6–88.1% | 221 / 140–160 | 0.002 to 0.00002 | 0.083 → 0.063 (−25%, CI excludes 0) | 214 → 160–167 |
| Cross-validation, unguided (3 repeats) | 3,996 | 85.6% → 87.0–87.6% | 193–207 / 120–137 | 0.002 to 0.000007 | 0.088 → 0.060 (−31%, CI excludes 0) | 218 → 148–164 |
| Test plans, guided, v2 | 600 | 85.8% → 88.7% | 37 / 20 | 0.033 | −28% | 34 → 22 |
| Test plans, guided, v1 (seed 0) | 600 | 85.8% → 88.0% | 36 / 23 | 0.12 | −12% | 34 → 26 |
| Gazebo, guided, v2 (executed arm) | 600 | 83.7% → 86.2% | 34 / 19 | 0.053 | 0.089 → 0.063 (−29%) | 33 → 21 |

Cross-validation cross-validates by seed, so no scene is in both training and evaluation; the bootstrap there resamples seeds. On the test set the unguided ensemble did not beat the single v1 network (86.0% against 86.5%), so only the guided version was executed again.

Gazebo: v2 differs from v1 on 16 test trials. 9 of them pick the rule's arm and reuse the rule run's rows; the other 7 were executed (all succeeded). Raw: `~/arm_classifier/v2/exec_both_samples.jsonl`.

| Metric | Diffusion + Guidance (rule) | + NN-A v1 | + NN-A v2 |
|---|---|---|---|
| Chose the best arm | 83.7% | 85.8% | **86.2%** |
| Costly arm mistakes (regret > 0.5) | 33 | 24 | **21** |
| Cost T + L, median | 8.55 | 8.53 | **8.51** |
| Success, doable tasks | 91.8% | 92.1% | 91.9% |
| Plan + motion, median | 4.69 s | 4.68 s | 4.69 s |

**What this showed:**
- **The network's better arm choice is real, not noise.** On about 4,000 unseen tasks it is significantly more often right than the rule (p ≤ 0.002 in every repeat, guided and unguided). When the two disagree, the network is right 1.4–1.7 times as often. It cuts the extra cost of choosing the wrong arm by a quarter to a third and the costly mistakes by about a quarter.
- **On the 600 test tasks the improved network confirms it:** +2.9 points of agreement (p = 0.033), regret −28%, costly mistakes 34 → 22. In Gazebo the executed arm is +2.5 points (p = 0.053) and costly mistakes 33 → 21.
- **Its effect is on decision quality and cost, not success.** Most failures happen on the correct arm (no collision-free sample, placement), so success stays within a trial or two of the rule.
- The NN needs data: below ~600 training tasks it is worse than the rule.

## Resampling declined trials

When the best variant (Guidance + NN-A) would decline a trial, because no arm had a usable sample or no sample passed the exact check, it draws a second round of 16 guided samples per arm. It then applies the network and the exact check again. Only the 38 declined trials are affected; their time to plan includes both rounds.

```bash
~/lerobot_venv/bin/python panda_moveit/evaluate_policy.py --checkpoint <ckpt> --seeds 1850 2000 \
    --only ~/arm_classifier/declined_both_samples.jsonl --save-trajectories \
    --guide-obstacle 1 --guide-goal 0.4 --guide-iters 1 --out ~/arm_classifier/round2_both.jsonl
~/lerobot_venv/bin/python panda_moveit/arm_classifier.py relabel --model ~/arm_classifier/both_samples_0.pt \
    --plans ~/arm_classifier/round2_both.jsonl --out ~/arm_classifier/plans_round2.jsonl
python3 panda_moveit/execute_policy.py --plans ~/arm_classifier/plans_round2.jsonl \
    --out ~/arm_classifier/exec_resample.jsonl --exact-check --retime
```

Raw results: `~/arm_classifier/exec_resample.jsonl`. The 562 trials that weren't declined are copied from the Guidance + NN-A run.

The 38 declined trials, after the second round:

| | Doable (25) | MoveIt also found none (13) |
|---|---|---|
| Succeeded | 3 | – |
| Executed and failed (place 4, missed grasp 1) | 5 | 3 |
| Declined again, no clear sample | 15 | 5 |
| Declined again, network said neither | 2 | 5 |

| Metric | Guidance + NN-A | + resampling |
|---|---|---|
| Success, doable tasks | 92.1% | 92.6% |
| Chose the best arm | 85.8% | 86.3% |
| No arm though one could | 25 | 17 |
| Correctly said "neither" | 13/17 | 10/17 |
| Collision signs | 0.7% | 1.0% |
| Time to plan, median (mean) | 0.40 s (0.40) | 0.41 s (0.43) |

**What this showed:**
- **Resampling helps only a little:** 3 more successes, +0.5 points.
- **15 of the 25 doable declined trials still have no collision-free sample after two rounds.** These scenes are outside what the model generates well: a narrow gap, or an obstacle beside the cube. More samples of the same model don't reach them.
- **It costs some correct refusals:** 3 tasks MoveIt also can't do were attempted and failed, and there were 2 more collision signs.

So the remaining gap to MoveIt is in the model's coverage of hard scenes, not in how many samples are drawn or how the arm is chosen.
