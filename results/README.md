# Results

Every run uses the same 600 test tasks: seeds 1850–1999 × 4 tasks. The diffusion model never saw these seeds in training, and the settings were not tuned on them.

| File | Run |
|---|---|
| [moveit_baseline.md](moveit_baseline.md) | MoveIt: plan both arms, execute the cheaper |
| [diffusion_original.md](diffusion_original.md) | Diffusion policy, first execution |
| [diffusion_fixed.md](diffusion_fixed.md) | Diffusion with the exact collision check and retiming |
| [diffusion_guided.md](diffusion_guided.md) | Fixed run plus guided sampling (obstacle and goal guidance during denoising) |

## Summary

| | Diffusion (original) | Diffusion (fixed) | Diffusion (guided) | MoveIt |
|---|---|---|---|---|
| Time to plan, median | 0.26 s | 0.27 s | 0.40 s | 0.88 s |
| Plan + motion, median (p90) | 6.05 s (13.0) | 4.62 s (5.9) | 4.69 s (5.8) | 5.22 s (6.5) |
| Cost T + L, median | 10.31 | 8.80 | 8.55 | 8.91 |
| Success, doable tasks | 86.1% | 85.9% | **91.8%** | 97.8% |
| Collision signs | 11.8% | 0.5% | 0.9% | planned collision-free |
| Chose the best arm | 83.5% | 83.2% | 83.7% | 95.0% |
| Correctly said "neither" | 8/17 | 15/17 | 13/17 | 17/17 |

To reproduce the side-by-side table:

```bash
python3 panda_moveit/compare_methods.py --policy Diffusion=~/exec_test_200k.jsonl \
    "Diffusion (fixed)"=~/exec_test_fixed.jsonl "Diffusion (guided)"=~/exec_test_guided.jsonl \
    --moveit ~/moveit_test.jsonl
```
