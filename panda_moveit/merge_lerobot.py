#!/usr/bin/env python3
"""Merge convert_lerobot.py outputs into one set of datasets.

New seeds (yours or a teammate's) are converted on their own and merged
with what is already converted, instead of reconverting everything:

    ~/lerobot_venv/bin/python convert_lerobot.py --data ~/panda_data --seeds 10000 1000 \\
        --out ~/lerobot_parts/seeds_10000 --height-map
    ~/lerobot_venv/bin/python merge_lerobot.py ~/lerobot_data ~/lerobot_parts/seeds_10000 \\
        --out ~/lerobot_merged

Each split (train/val/test) is merged from the inputs that have it, in the
order given. The inputs must share their conversion settings and must not
share seeds. Writes the same layout as convert_lerobot.py: <out>/<split>,
<split>_episodes.jsonl (episode indices renumbered), infeasible.jsonl and
conversion.json (with the list of inputs).
"""
import argparse
import json
import shutil
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

from convert_lerobot import SPLITS

SPLIT_NAMES = (*SPLITS, 'train')
# conversion.json keys that must match for the episodes to mean the same thing.
SETTINGS = ('fps', 'arms', 'splits', 'image_size', 'second_arm_image_rotated_180', 'height_map',
            'gripper', 'bases', 'environment_state_names', 'action')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inputs', nargs='+', help='convert_lerobot.py (or merge_lerobot.py) output folders')
    p.add_argument('--out', required=True)
    p.add_argument('--repo-prefix', default='dual_panda', help='repo_id is <prefix>_<split>')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    inputs = [Path(i).expanduser() for i in args.inputs]
    out = Path(args.out).expanduser()
    if out in inputs:
        raise SystemExit('--out must be a new folder, not one of the inputs')
    configs = [json.loads((i / 'conversion.json').read_text()) for i in inputs]
    for key in SETTINGS:
        for i, config in zip(inputs, configs):
            if config.get(key) != configs[0].get(key):
                raise SystemExit(f'{key} differs between {inputs[0]} and {i}')

    # Every (trial, arm) is either an episode or infeasible, so these cover every seed.
    seeds = {}
    for i in inputs:
        lines = read_jsonl(i / 'infeasible.jsonl')
        for split in SPLIT_NAMES:
            lines += read_jsonl(i / f'{split}_episodes.jsonl')
        for seed in {line['seed'] for line in lines}:
            if seed in seeds:
                raise SystemExit(f'seed {seed} is in both {seeds[seed]} and {i}')
            seeds[seed] = i

    if out.exists():
        if not args.overwrite:
            raise SystemExit(f'{out} exists; pass --overwrite to replace it')
        shutil.rmtree(out)
    out.mkdir(parents=True)

    counts = {}
    for split in SPLIT_NAMES:
        parts = [i for i in inputs if (i / split).exists()]
        if not parts:
            continue
        aggregate_datasets(repo_ids=[f'{args.repo_prefix}_{split}'] * len(parts),
                           aggr_repo_id=f'{args.repo_prefix}_{split}',
                           roots=[i / split for i in parts], aggr_root=out / split)
        offset = 0
        with (out / f'{split}_episodes.jsonl').open('w') as log:
            for i in parts:
                lines = read_jsonl(i / f'{split}_episodes.jsonl')
                for line in lines:
                    log.write(json.dumps({**line, 'episode_index': line['episode_index'] + offset}) + '\n')
                offset += len(lines)
        counts[split] = offset
    with (out / 'infeasible.jsonl').open('w') as f:
        for i in inputs:
            for line in read_jsonl(i / 'infeasible.jsonl'):
                f.write(json.dumps(line) + '\n')

    (out / 'conversion.json').write_text(json.dumps(dict(
        configs[0], source=None, seeds=None,
        merged=[dict(folder=str(i), source=c.get('source'), seeds=c.get('seeds'), trials=c['trials'],
                     episodes=c['episodes']) for i, c in zip(inputs, configs)],
        trials=sum(c['trials'] for c in configs), episodes=counts), indent=2))
    print(f'Done: {len(seeds)} seeds, episodes {counts} in {out}')


if __name__ == '__main__':
    main()
