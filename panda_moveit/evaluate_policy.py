#!/usr/bin/env python3
"""Evaluate a trained Diffusion Policy checkpoint against the MoveIt labels.

For each trial (seed, task): sample K trajectories per arm (both arms in one
batch, from the before image and the start state), roll out a second chunk
if a sample hasn't finished, then score each sample with forward kinematics
and the approximate collision check:
    valid = grasp and release found, TCP within GOAL_TOLERANCE of the cube
            at the grasp and of the place target at the release, no collision
The arm with the cheapest valid sample is assigned (the fixed-formula
baseline); that is compared with MoveIt's assignment.

Run with the LeRobot venv:
    ~/lerobot_venv/bin/python evaluate_policy.py \
        --checkpoint outputs/diffusion_dual_panda/checkpoints/last/pretrained_model \
        --seeds 1700 1850 --samples 16 --out ~/eval_val.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

from collision_check import check_many
from convert_lerobot import GRIPPER_CLOSE, GRIPPER_OPEN, SceneImages, environment_state
from panda_kinematics import tcp_position

ARMS = ('first', 'second')
GRIPPER_MID = (GRIPPER_OPEN + GRIPPER_CLOSE) / 2
GOAL_TOLERANCE = 0.03   # m; enough for cost estimation, not for execution
STILL = 0.005           # rad per frame: below this the arm counts as stopped
MAX_CHUNKS = 2
LAMBDA = 1.0            # pick_place.LAMBDA


class Sampler:
    def __init__(self, checkpoint, device, bf16):
        self.policy = DiffusionPolicy.from_pretrained(checkpoint).to(device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=checkpoint)
        self.image_keys = list(self.policy.config.image_features)
        self.device = device
        self.bf16 = bf16 and device == 'cuda'

    @torch.no_grad()
    def chunk(self, states, envs, images, repeat=1):
        """Action chunks for observation rows: states (B, 8), envs (B, 71),
        images {key: (B, H, W, 3) uint8}. Each row is encoded once and
        sampled `repeat` times -> (B * repeat, horizon, 8) numpy, rows grouped."""
        batch = {'observation.state': torch.as_tensor(states, dtype=torch.float32),
                 'observation.environment_state': torch.as_tensor(envs, dtype=torch.float32)}
        for key in self.image_keys:
            batch[key] = torch.as_tensor(images[key]).permute(0, 3, 1, 2).float() / 255.0
        batch = self.pre(batch)
        for key in ['observation.state', 'observation.environment_state'] + self.image_keys:
            batch[key] = batch[key].unsqueeze(1)  # (B, n_obs_steps=1, ...)
        batch['observation.images'] = torch.stack([batch[k] for k in self.image_keys], dim=-4)
        model = self.policy.diffusion
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.bf16):
            # Encoding (two ResNets per row) is the fixed cost: do it once per
            # row, not once per sample, then denoise all samples together.
            cond = model._prepare_global_conditioning(batch).repeat_interleave(repeat, 0)
            actions = model.conditional_sample(cond.shape[0], global_cond=cond)
        actions = actions[:, :self.policy.config.n_action_steps].float()  # n_obs_steps = 1
        return self.post(actions).cpu().numpy()


def rollout(sampler, start, env, images, samples):
    """K samples per arm, all arms in one batch per chunk. start/env/images
    are dicts by arm; returns {arm: list of (frames, 8) arrays}."""
    arms = list(start)
    trajs = {(a, k): [np.asarray(start[a], np.float32)] for a in arms for k in range(samples)}
    # First chunk: every sample of an arm has the same observation.
    chunks = sampler.chunk(np.stack([start[a] for a in arms]), np.stack([env[a] for a in arms]),
                           {k: np.stack([images[a][k] for a in arms]) for k in sampler.image_keys},
                           repeat=samples)
    for key, chunk in zip(trajs, chunks):
        trajs[key].extend(chunk)
    active = [key for key in trajs if not finished(np.array(trajs[key]))]
    for _ in range(MAX_CHUNKS - 1):
        if not active:
            break
        states = np.stack([trajs[key][-1] for key in active])
        envs = np.stack([env[a] for a, _ in active])
        imgs = {k: np.stack([images[a][k] for a, _ in active]) for k in sampler.image_keys}
        chunks = sampler.chunk(states, envs, imgs)
        for key, chunk in zip(active, chunks):
            trajs[key].extend(chunk)
        active = [key for key in active if not finished(np.array(trajs[key]))]
    return {a: [np.array(trajs[(a, k)]) for k in range(samples)] for a in arms}


def finished(traj):
    closed = np.flatnonzero(traj[:, 7] < GRIPPER_MID)
    if not len(closed) or not (traj[closed[0]:, 7] > GRIPPER_MID).any():
        return False
    return np.abs(np.diff(traj[-3:, :7], axis=0)).max() < STILL


def motion_end(traj):
    """Frame count up to the last frame where the arm still moves."""
    moving = np.flatnonzero(np.abs(np.diff(traj[:, :7], axis=0)).max(axis=1) > STILL)
    return moving[-1] + 2 if len(moving) else 1


def score_arm(trajs, arm, scenario, trial, fps):
    """Score all samples of one arm; one vectorized collision check."""
    ends = [motion_end(t) for t in trajs]
    collisions = check_many([t[:e, :7] for t, e in zip(trajs, ends)], arm, scenario)
    return [score(t, e, c, arm, trial, fps) for t, e, c in zip(trajs, ends, collisions)]


def score(traj, end, collision, arm, trial, fps):
    """Measurements of one sampled trajectory (frames, 8), moving for end frames."""
    q = traj[:end, :7]
    T = (end - 1) / fps
    L = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
    r = dict(T=T, L=L, cost=T + LAMBDA * L, frames=int(end))
    closed = np.flatnonzero(traj[:, 7] < GRIPPER_MID)
    grasp = int(closed[0]) if len(closed) else None
    opened = np.flatnonzero(traj[grasp:, 7] > GRIPPER_MID) + grasp if grasp is not None else []
    release = int(opened[0]) if len(opened) else None
    tcp = tcp_position(traj[:, :7], arm)

    def error(change, target):
        # At 5 Hz no frame sits exactly where the gripper acts: it changes
        # between the frame before and the frame of the switch, so use the closer one.
        if change is None:
            return None
        frames = tcp[max(change - 1, 0):change + 1]
        return float(np.linalg.norm(frames - np.asarray(target), axis=1).min())
    r['grasp_error'] = error(grasp, trial['cube_pose']['xyz'])
    r['place_error'] = error(release, trial['place_xyz'])
    r.update(clearance=collision['clearance'], collision=collision['collision'],
             collision_object=collision['object'] if collision['collision'] else None)
    r['valid'] = bool(release is not None and not r['collision']
                      and r['grasp_error'] < GOAL_TOLERANCE and r['place_error'] < GOAL_TOLERANCE)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default=str(Path.home() / 'panda_data'))
    p.add_argument('--seeds', type=int, nargs=2, default=(1700, 1850), metavar=('FIRST', 'END'))
    p.add_argument('--samples', type=int, default=16, help='K per arm')
    p.add_argument('--max-trials', type=int)
    p.add_argument('--fps', type=int, default=5)
    p.add_argument('--no-bf16', action='store_true', help='sample in float32')
    p.add_argument('--out', default=str(Path.home() / 'eval.jsonl'))
    args = p.parse_args()

    data = Path(args.data).expanduser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sampler = Sampler(args.checkpoint, device, bf16=not args.no_bf16)
    height = 'observation.images.height' in sampler.image_keys
    trials = [json.loads(l) for l in (data / 'trials.jsonl').read_text().splitlines() if l]
    trials = [t for t in trials if args.seeds[0] <= t['seed'] < args.seeds[1]]
    trials.sort(key=lambda t: (t['seed'], t['task_index']))
    trials = trials[:args.max_trials] if args.max_trials else trials

    rows, out = [], Path(args.out).expanduser().open('w')
    for n, summary in enumerate(trials, 1):
        trial = json.loads((data / summary['trial']).read_text())
        scenario = json.loads((data / trial['scenario']).read_text())
        scene = SceneImages(data, trial, height)
        start = {a: list(scenario['arm_starts'][a]) + [GRIPPER_OPEN] for a in ARMS}
        env = {a: environment_state(scenario, trial, a) for a in ARMS}
        images = {a: {'observation.images.overhead': scene.view(scene.rgb, a),
                      **({'observation.images.height': scene.view(scene.height, a)} if height else {})}
                  for a in ARMS}

        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        samples = rollout(sampler, start, env, images, args.samples)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        scores = {a: score_arm(samples[a], a, scenario, trial, args.fps) for a in ARMS}
        t2 = time.perf_counter()

        row = dict(seed=trial['seed'], task_index=trial['task_index'],
                   sample_time=t1 - t0, check_time=t2 - t1, arms={})
        for a in ARMS:
            valid = [s for s in scores[a] if s['valid']]
            best = min(valid, key=lambda s: s['cost'], default=None)
            row['arms'][a] = dict(
                moveit_feasible=trial['arms'][a]['feasible'], moveit_cost=trial['arms'][a]['best_cost'],
                valid_fraction=len(valid) / len(scores[a]), predicted_cost=best and best['cost'],
                samples=scores[a])
        costs = {a: row['arms'][a]['moveit_cost'] for a in ARMS if row['arms'][a]['moveit_feasible']}
        pred = {a: row['arms'][a]['predicted_cost'] for a in ARMS if row['arms'][a]['predicted_cost'] is not None}
        row['moveit_arm'] = min(costs, key=costs.get) if costs else None
        row['policy_arm'] = min(pred, key=pred.get) if pred else None
        row['regret'] = (costs[row['policy_arm']] - costs[row['moveit_arm']]
                         if row['policy_arm'] in costs and row['moveit_arm'] else None)
        out.write(json.dumps(row) + '\n')
        rows.append(row)
        if n % 25 == 0 or n == len(trials):
            print(f'{n}/{len(trials)} trials', flush=True)
    out.close()
    summarize(rows)


def summarize(rows):
    n = len(rows)
    agree = sum(r['policy_arm'] == r['moveit_arm'] for r in rows)
    regrets = [r['regret'] for r in rows if r['regret'] is not None]
    wrong_arm = sum(r['policy_arm'] is not None and not r['arms'][r['policy_arm']]['moveit_feasible'] for r in rows)
    samples = [s for r in rows for a in ARMS for s in r['arms'][a]['samples']]
    feasible_samples = [s for r in rows for a in ARMS if r['arms'][a]['moveit_feasible']
                        for s in r['arms'][a]['samples']]

    def frac(xs, f):
        return sum(f(x) for x in xs) / max(len(xs), 1)

    print(f'\n== {n} trials, {len(samples)} samples')
    print(f'assignment agrees with MoveIt: {agree}/{n} ({100 * agree / n:.1f}%)')
    if regrets:
        print(f'regret (MoveIt cost of chosen arm - best): mean {np.mean(regrets):.3f}  '
              f'median {np.median(regrets):.3f}  >0.5: {sum(r > 0.5 for r in regrets)}')
    print(f'assigned an arm MoveIt found infeasible: {wrong_arm}')
    print(f'samples valid (arms MoveIt can do): {100 * frac(feasible_samples, lambda s: s["valid"]):.1f}%  '
          f'collision-free: {100 * frac(feasible_samples, lambda s: not s["collision"]):.1f}%')
    ge = [s['grasp_error'] for s in feasible_samples if s['grasp_error'] is not None]
    pe = [s['place_error'] for s in feasible_samples if s['place_error'] is not None]
    if ge:
        print(f'grasp error median {100 * np.median(ge):.1f} cm, place error median '
              f'{100 * np.median(pe) if pe else float("nan"):.1f} cm')
    pairs = [(r['arms'][a]['predicted_cost'], r['arms'][a]['moveit_cost']) for r in rows for a in ARMS
             if r['arms'][a]['predicted_cost'] is not None and r['arms'][a]['moveit_cost'] is not None]
    if pairs:
        d = np.array([p - m for p, m in pairs])
        print(f'best valid sample cost - MoveIt cost: mean {d.mean():.3f}  median {np.median(d):.3f}')
    st = [r['sample_time'] for r in rows]
    ct = [r['check_time'] for r in rows]
    print(f'decision time: sampling median {1000 * np.median(st):.0f} ms + checks {1000 * np.median(ct):.0f} ms '
          f'(MoveIt both arms ~940 ms)')


if __name__ == '__main__':
    main()
