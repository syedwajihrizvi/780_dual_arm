#!/usr/bin/env python3
"""Summary statistics of a collect.py dataset: feasibility, which arm wins,
cost gaps, label noise across repeats, and the MoveIt planning-time baseline.

    data_stats.py                       # ~/panda_data
    data_stats.py --data ~/panda_data --seeds 0 1700
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

BASES = {'first': (0.2, 0.0), 'second': (1.5, 0.0)}
ARMS = ('first', 'second')


def pct(n, total):
    return f'{n} ({100 * n / total:.1f}%)' if total else '0'


def binomial_p(k, n):
    """Two-sided p-value for k of n under a fair coin (normal approximation)."""
    if n == 0:
        return 1.0
    z = abs(k - n / 2) / math.sqrt(n / 4)
    return math.erfc(z / math.sqrt(2))


def percentiles(values, ps=(50, 90, 99)):
    return '  '.join(f'p{p} {np.percentile(values, p):.2f}' for p in ps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(Path.home() / 'panda_data'))
    p.add_argument('--seeds', type=int, nargs=2, metavar=('FIRST', 'END'), help='seed range [FIRST, END)')
    args = p.parse_args()
    data = Path(args.data).expanduser()

    trials = [json.loads(line) for line in (data / 'trials.jsonl').read_text().splitlines() if line]
    if args.seeds:
        trials = [t for t in trials if args.seeds[0] <= t['seed'] < args.seeds[1]]
    n = len(trials)
    seeds = {t['seed'] for t in trials}
    print(f'== Dataset: {len(seeds)} seeds, {n} paired trials ({n / len(seeds):.1f} per seed)\n')

    # -- Feasibility
    feasible = {a: sum(t['arms'][a]['feasible'] for t in trials) for a in ARMS}
    both = [t for t in trials if all(t['arms'][a]['feasible'] for a in ARMS)]
    neither = sum(not any(t['arms'][a]['feasible'] for a in ARMS) for t in trials)
    only = {a: sum(t['arms'][a]['feasible'] and not t['arms'][o]['feasible'] for t in trials)
            for a, o in (('first', 'second'), ('second', 'first'))}
    print('== Feasibility')
    for a in ARMS:
        print(f'  {a:6} arm feasible: {pct(feasible[a], n)}')
    print(f'  both: {pct(len(both), n)}   only first: {pct(only["first"], n)}   '
          f'only second: {pct(only["second"], n)}   neither: {pct(neither, n)}')
    episodes = feasible['first'] + feasible['second']
    print(f'  -> {episodes} policy episodes ({episodes / n:.2f} per trial)\n')

    # -- Assignment (the label)
    def winner(t):
        costs = {a: t['arms'][a]['best_cost'] for a in ARMS if t['arms'][a]['feasible']}
        return min(costs, key=costs.get) if costs else None

    wins = Counter(winner(t) for t in trials)
    print('== Assignment (lowest best cost among feasible arms)')
    print(f'  first: {pct(wins["first"], n)}   second: {pct(wins["second"], n)}   none: {pct(wins[None], n)}')
    k = sum(winner(t) == 'second' for t in both)
    print(f'  when both feasible: second cheaper in {pct(k, len(both))}, '
          f'p = {binomial_p(k, len(both)):.3f} vs a fair coin')

    gaps = np.array([abs(t['arms']['first']['best_cost'] - t['arms']['second']['best_cost']) for t in both])
    print(f'  cost gap |C1 - C2| when both feasible: {percentiles(gaps, (10, 25, 50, 75, 90))}')
    for tie in (0.25, 0.5, 1.0):
        print(f'    gap < {tie}: {pct(int((gaps < tie).sum()), len(both))}')

    def nearest(t):
        x, y = t['cube_pose']['xyz'][:2]
        return min(ARMS, key=lambda a: math.dist((x, y), BASES[a]))
    agree = sum(nearest(t) == winner(t) for t in both)
    print(f'  nearest arm is the cheaper one (both feasible): {pct(agree, len(both))}\n')

    # -- Costs
    for key, label in (('best_cost', 'cost C'), ('best_trajectory_duration', 'time T (s)'),
                       ('best_joint_path_length', 'path L (rad)')):
        v = [t['arms'][a][key] for t in trials for a in ARMS if t['arms'][a]['feasible']]
        print(f'== {label:12}: mean {np.mean(v):.2f}  {percentiles(v, (5, 50, 95))}')
    print()

    # -- Full records: label noise across repeats, why plans fail
    spread, best_index, failed = [], Counter(), Counter()
    for t in trials:
        record = json.loads((data / t['trial']).read_text())
        for a in ARMS:
            arm = record['arms'][a]
            costs = [pl['cost'] for pl in arm['plans'] if pl['success']]
            if len(costs) >= 2:
                spread.append(max(costs) - min(costs))
            if arm['feasible']:
                best_index[arm['best_plan_index']] += 1
            else:
                failed[arm['plans'][-1]['failed_stage']] += 1
    print('== Label noise (repeats of the same task)')
    print(f'  cost spread best-to-worst feasible repeat: {percentiles(spread, (25, 50, 75, 90))}')
    total = sum(best_index.values())
    print('  best plan was repeat ' + ', '.join(f'#{i}: {pct(c, total)}' for i, c in sorted(best_index.items())))
    print('  infeasible arms, stage that failed: ' +
          ', '.join(f'{s}: {pct(c, sum(failed.values()))}' for s, c in failed.most_common()))
    print()

    # -- Planning-time baseline
    single = [t['arms'][a]['single_plan_wall_time'] for t in trials for a in ARMS]
    single_feasible = [t['arms'][a]['single_plan_wall_time'] for t in trials for a in ARMS
                       if t['arms'][a]['feasible']]
    decision = [sum(t['arms'][a]['single_plan_wall_time'] for a in ARMS) for t in trials]
    print('== MoveIt planning time (the baseline to beat)')
    print(f'  one plan, one arm (all):      {percentiles(single)} s')
    print(f'  one plan, one arm (feasible): {percentiles(single_feasible)} s')
    print(f'  decision = plan both arms:    mean {np.mean(decision):.2f}  {percentiles(decision)} s')


if __name__ == '__main__':
    main()
