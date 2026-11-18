# Results

Every run uses the same 600 test tasks: seeds 1850–1999 × 4 tasks. The diffusion model never saw these seeds in training. Sampler, guidance and network settings were chosen on the validation seeds (1700–1849) or by cross-validation on a teammate's seeds (2000–2999). One rule was added after looking at test plans: the network chooses only among arms that have a usable sample (see [diffusion_nn.md](diffusion_nn.md)).

| File | Run |
|---|---|
| [moveit_baseline.md](moveit_baseline.md) | MoveIt: plan both arms, execute the cheaper |
| [diffusion_original.md](diffusion_original.md) | Diffusion policy, first execution |
| [diffusion_fixed.md](diffusion_fixed.md) | Diffusion with the exact collision check and retiming ("Diffusion" below) |
| [diffusion_guided.md](diffusion_guided.md) | Plus guided sampling (obstacle and goal guidance during denoising) |
| [diffusion_nn.md](diffusion_nn.md) | Plus a learned arm choice (NN-A: sample features; NN-B: plus scene features), and resampling declined trials |
| [corrections_and_fast_sampler.md](corrections_and_fast_sampler.md) | **Read this first.** Two measurement fixes (MoveIt's planning time, the exact check), the MoveIt fallback and the faster sampler. Its numbers replace the earlier files' where they differ. |

## Summary: final system (faster sampler)

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

- **Speed:** diffusion plans in 0.10 s without guidance and 0.16 s with it, against 0.38 s for MoveIt planning the arms one after the other and an estimated 0.20 s in parallel.
- **Reliability:** 93.8% alone. With MoveIt as a fallback for the 28 tasks the policy declines, 96.2%, which is not statistically different from MoveIt's 97.8% on these 600 tasks (McNemar p = 0.75), at the same median time to plan.
- **Cost:** the policy's motions are cheaper than MoveIt's (8.52 against 8.91).
- **Guidance:** success on doable tasks rises from 86.6% to 93.8% (59 tasks gained, 13 lost; McNemar p < 0.001), for 0.06 s more planning.
- **Learned arm choice:** significantly better than the rule in cross-validation on 3,996 tasks (p ≤ 0.002). On the test set it is significantly better without guidance (best arm 84.0% → 86.8%, p = 0.027 for NN-A and 0.033 for NN-B) and not detectably better with guidance (p = 1.0 for NN-A, 0.22 for NN-B). It doesn't change success.

## Earlier sampler (corrected)

These are the runs of the earlier files, with the exact-check fix applied and MoveIt's planning time re-measured.

| | Diffusion | Diffusion + NN-A | Diffusion + NN-B | Diffusion + Guidance | Guidance + NN-A | Guidance + NN-B | Guidance + NN-A v2 | Guidance + NN-A + resampling | Guidance + NN-A + MoveIt fallback |
|---|---|---|---|---|---|---|---|---|---|
| Time to plan, median | 0.27 s | 0.27 s | 0.27 s | 0.40 s | 0.40 s | 0.40 s | 0.40 s | 0.41 s | 0.41 s |
| Plan + motion, median (p90) | 4.62 s (5.89) | 4.62 s (5.90) | 4.62 s (5.89) | 4.69 s (5.85) | 4.68 s (5.87) | 4.69 s (5.85) | 4.69 s (5.85) | 4.69 s (5.88) | 4.70 s (6.03) |
| Cost T + L, median | 8.81 | 8.78 | 8.78 | 8.55 | 8.53 | 8.53 | 8.51 | 8.54 | 8.58 |
| Success, doable tasks | 86.3% | 86.4% | 86.1% | 92.5% | 92.5% | 92.5% | 92.5% | 92.8% | 95.7% |
| Collision signs | 0.5% | 0.5% | 0.5% | 0.9% | 0.7% | 0.7% | 0.9% | 1.0% | 0.7% |
| Chose the best arm | 83.5% | 85.0% | 84.8% | 83.8% | 86.0% | 85.2% | 86.3% | 86.3% | 89.0% |
| Correctly said "neither" | 15/17 | 15/17 | 15/17 | 13/17 | 13/17 | 13/17 | 13/17 | 10/17 | 12/17 |

The first execution, before the exact check and retiming, is in [diffusion_original.md](diffusion_original.md): 86.1% success and 11.8% collision signs.

## Notes

- **MoveIt's time to plan** covers both arms planned one after the other, as `execute_moveit.py` runs it. The earlier figure of 0.88 s included a polling delay in our own code and is wrong. The parallel column takes, per task, the slower arm's planning time; it slightly favours MoveIt, since two planners running at once would share the CPU.
- **Best arm** means agreement with the labels (the cheaper arm over 3 MoveIt planning repeats per arm). Two single MoveIt runs agree with each other on 93.3% of tasks, and about a third of the tasks both arms can do are near-ties, so 100% is not reachable.
- **Resampling:** when the policy would decline a trial, it draws a second round of 16 guided samples per arm and decides again.
- **MoveIt fallback:** when the policy declines a trial, MoveIt plans and executes it; its planning time is added to that trial's time to plan.

To reproduce the side-by-side tables:

```bash
python3 panda_moveit/compare_methods.py --policy Diffusion=~/eval_fast/exec_unguided.jsonl \
    "Diffusion+NN-A"=~/eval_fast/exec_nnA_unguided.jsonl "Diffusion+NN-B"=~/eval_fast/exec_nnB_unguided.jsonl \
    Guidance=~/eval_fast/exec_guided.jsonl "Guidance+NN-A"=~/eval_fast/exec_nn.jsonl \
    "Guidance+NN-B"=~/eval_fast/exec_nnB_guided.jsonl "+resampling"=~/eval_fast/exec_resample.jsonl \
    "+fallback"=~/eval_fast/exec_fallback.jsonl --moveit ~/moveit_timing/moveit_test_retimed.jsonl
# earlier sampler: the files in ~/exec_fix/; parallel MoveIt: --moveit ~/moveit_timing/moveit_test_retimed_parallel.jsonl
```
