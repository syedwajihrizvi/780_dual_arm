#!/usr/bin/env python3
"""Side-by-side table of policy runs (execute_policy.py) and MoveIt
(execute_moveit.py) on the trials all of them have run.

    python3 compare_methods.py --policy Diffusion=~/exec_test_200k.jsonl \
        "Diffusion (fixed)"=~/exec_test_fixed.jsonl --moveit ~/moveit_test.jsonl

Cost is T + L of what was executed: for the policy the commanded duration
(after --retime, if used) plus the sample's joint path length; for MoveIt the
executed plan's cost.

Doable trials are those where the labels (the best of the repeated plans in
the data) have an arm; the success rate is over those, so a method that
correctly says "neither" on an impossible task is not penalised.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    return {(r['seed'], r['task_index']): r
            for r in map(json.loads, Path(path).expanduser().read_text().splitlines()) if r}


def stats(values):
    v = np.array([x for x in values if x is not None], dtype=float)
    if not len(v):
        return '-'
    return f'{np.median(v):.2f} (mean {v.mean():.2f}, p90 {np.percentile(v, 90):.2f})'


def pct(k, n):
    return f'{k}/{n} ({100 * k / max(n, 1):.1f}%)'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', nargs='+', required=True, metavar='[NAME=]PATH',
                   help='execute_policy.py outputs, one column each')
    p.add_argument('--moveit', required=True, help='execute_moveit.py output')
    p.add_argument('--data', default=str(Path.home() / 'panda_data'))
    args = p.parse_args()

    policies = {}
    for spec in args.policy:
        name, _, path = spec.rpartition('=')
        policies[name or Path(path).stem] = load(path)
    moveit = load(args.moveit)
    keys = sorted(set(moveit).intersection(*policies.values()))
    data = Path(args.data).expanduser()
    feasible = {}
    for line in (data / 'trials.jsonl').read_text().splitlines():
        t = json.loads(line)
        if (t['seed'], t['task_index']) in moveit:
            feasible[(t['seed'], t['task_index'])] = {a: v['feasible'] for a, v in t['arms'].items()}

    # Normalise both to: arm, label, decision, motion, wall, success, failed
    rows = {name: [] for name in [*policies, 'MoveIt']}
    for k in keys:
        for name, policy in policies.items():
            d = policy[k]
            # The sample's cost is T + L at its 5 Hz timing; swap in the executed T.
            sampled = d.get('sampled_duration', d.get('policy_duration'))
            cost = (d['policy_cost'] - sampled + d['policy_duration']
                    if d.get('policy_cost') is not None and d.get('policy_duration') is not None else None)
            rows[name].append(dict(arm=d['policy_arm'], label=d['moveit_arm'], decision=d['policy_decision_time'],
                                   motion=d.get('policy_duration'), wall=d.get('duration'), cost=cost,
                                   success=bool(d.get('success')), failed=d.get('failed_stage'),
                                   collision=bool(d.get('moveit_contacts')) or d.get('tracking_error', 0) > 0.15,
                                   feasible=feasible[k]))
        m = moveit[k]
        rows['MoveIt'].append(dict(arm=m['arm'], label=m['label_arm'], decision=m['decision_time'],
                                   motion=m.get('motion_time'), wall=m.get('execute_wall_time'),
                                   cost=m.get('executed_cost'),
                                   success=bool(m.get('success')), failed=m.get('failed_stage'),
                                   collision=None, feasible=feasible[k]))

    n = len(keys)
    print(f'{n} trials in both ({len({k[0] for k in keys})} seeds)\n')
    table = []
    for name, rs in rows.items():
        doable = [r for r in rs if r['label']]
        done = [r for r in rs if r['success']]
        failures = {}
        for r in rs:
            if r['arm'] and not r['success']:
                failures[r['failed']] = failures.get(r['failed'], 0) + 1
        table.append((name, {
            'Time to plan, s (both arms)': stats(r['decision'] for r in rs),
            'Motion time, s (completed)': stats(r['motion'] for r in done),
            'Execution wall time, s (completed)': stats(r['wall'] for r in done),
            'Plan + motion, s (completed)': stats(r['decision'] + r['motion'] for r in done),
            'Cost T + L (completed)': stats(r['cost'] for r in done),
            'Success rate (doable tasks)': pct(sum(r['success'] for r in doable), len(doable)),
            'Success rate (all tasks)': pct(len(done), n),
            'Arm choice = best arm (labels)': pct(sum(r['arm'] == r['label'] for r in rs), n),
            'Chose an arm that cannot do it': pct(sum(bool(r['arm']) and not r['feasible'][r['arm']] for r in rs), n),
            'No arm though one could': pct(sum(r['arm'] is None and r['label'] is not None for r in rs), n),
            'Correctly said neither': f"{sum(r['arm'] is None and r['label'] is None for r in rs)}"
                                      f"/{sum(r['label'] is None for r in rs)}",
            'Collision signs (executed)': (pct(sum(bool(r['collision']) for r in rs if r['arm']),
                                               sum(bool(r['arm']) for r in rs))
                                           if name != 'MoveIt' else 'planned collision-free'),
            'Failures by stage':', '.join(f'{k} {v}' for k, v in sorted(failures.items(), key=lambda kv: -kv[1]))
                                 or '-',
        }))
    width = max(len(k) for k in table[0][1])
    cols = [max(len(name), *(len(t[k]) for k in t)) for name, t in table]
    print(' ' * width + ' | ' + ' | '.join(f'{name:<{c}}' for (name, _), c in zip(table, cols)))
    for key in table[0][1]:
        print(f'{key:<{width}} | ' + ' | '.join(f'{t[key]:<{c}}' for (_, t), c in zip(table, cols)))


if __name__ == '__main__':
    main()
