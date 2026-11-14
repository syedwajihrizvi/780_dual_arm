# Results

Every run uses the same 600 test tasks: seeds 1850–1999 × 4 tasks. The diffusion model never saw these seeds in training, and the settings were not tuned on them.

| File | Run |
|---|---|
| [moveit_baseline.md](moveit_baseline.md) | MoveIt: plan both arms, execute the cheaper |
| [diffusion_original.md](diffusion_original.md) | Diffusion policy, first execution |
| [diffusion_fixed.md](diffusion_fixed.md) | Diffusion with the exact collision check and retiming ("Diffusion" below) |
| [diffusion_guided.md](diffusion_guided.md) | Plus guided sampling (obstacle and goal guidance during denoising) |
| [diffusion_nn.md](diffusion_nn.md) | Plus a learned arm choice (NN-A: sample features; NN-B: plus scene features), and resampling declined trials |

## Summary

| | Diffusion | Diffusion + NN-A | Diffusion + NN-B | Diffusion + Guidance | Guidance + NN-A | Guidance + NN-B | Guidance + NN-A + resampling | MoveIt | MoveIt, arms planned in parallel |
|---|---|---|---|---|---|---|---|---|---|
| Time to plan, median | 0.27 s | 0.27 s | 0.27 s | 0.40 s | 0.40 s | 0.40 s | 0.41 s | 0.88 s | 0.53 s |
| Plan + motion, median (p90) | 4.62 s (5.89) | 4.62 s (5.90) | 4.62 s (5.89) | 4.69 s (5.82) | 4.68 s (5.85) | 4.69 s (5.84) | 4.69 s (5.88) | 5.22 s (6.46) | 4.78 s (6.18) |
| Cost T + L, median | 8.80 | 8.78 | 8.78 | 8.55 | 8.53 | 8.53 | 8.54 | 8.91 | 8.91 |
| Success, doable tasks | 85.9% | 86.3% | 85.8% | 91.8% | 92.1% | 91.9% | **92.6%** | 97.8% | 97.8% |
| Collision signs | 0.5% | 0.5% | 0.5% | 0.9% | 0.7% | 0.7% | 1.0% | planned collision-free | planned collision-free |
| Chose the best arm | 83.2% | 84.8% | 84.5% | 83.7% | 85.8% | 85.0% | **86.3%** | 95.0% | 95.0% |
| Correctly said "neither" | 15/17 | 15/17 | 15/17 | 13/17 | 13/17 | 13/17 | 10/17 | 17/17 | 17/17 |

The first execution, before the exact check and retiming, is in [diffusion_original.md](diffusion_original.md): 86.1% success and 11.8% collision signs.

MoveIt's time to plan covers both arms planned one after the other, as `execute_moveit.py` runs it. The parallel column takes, per task, the slower arm's share of that time; the labels record each arm's planning time on the same PC, and their sum matches this run (median 0.90 s against 0.88 s). It is a slight underestimate: two planners running at once would share the CPU. Raw: `~/moveit_test_parallel.jsonl`.

Resampling ([diffusion_nn.md](diffusion_nn.md#resampling-declined-trials)): when Guidance + NN-A would decline a trial, it draws a second round of 16 guided samples per arm and decides again.

To reproduce the side-by-side table:

```bash
python3 panda_moveit/compare_methods.py --policy Diffusion=~/exec_test_fixed.jsonl \
    "Diffusion+NN-A"=~/arm_classifier/exec_none_samples.jsonl "Diffusion+NN-B"=~/arm_classifier/exec_none_both.jsonl \
    "Diffusion+Guidance"=~/exec_test_guided.jsonl \
    "Guidance+NN-A"=~/arm_classifier/exec_both_samples.jsonl "Guidance+NN-B"=~/arm_classifier/exec_both_both.jsonl \
    "Guidance+NN-A+resample"=~/arm_classifier/exec_resample.jsonl --moveit ~/moveit_test.jsonl
# MoveIt with parallel planning: --moveit ~/moveit_test_parallel.jsonl
```
