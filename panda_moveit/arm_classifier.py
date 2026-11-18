#!/usr/bin/env python3
"""Learned arm assignment from the diffusion samples.

evaluate_policy.py assigns the arm with the cheapest valid sample. This
replaces that rule with a small network trained on MoveIt's labels: each arm's
K samples are summarized into SAMPLE_FEATURES, and the network outputs 3 logits,
(first, second, neither).

    shared  (default): one MLP scores each arm from its own features (same
            weights for both arms, which mirror each other); a learned bias
            is the "neither" logit, the score both arms must beat.
    plain:  one MLP on both arms' features concatenated -> 3 logits.

--features samples uses only the sample summaries; --features both adds
SCENE_FEATURES per arm (start pose, obstacles near the cube and target, the
other arm's position), to compare whether the samples already carry the scene.

The inputs are plan files written by evaluate_policy.py (the samples' scores
are in row['arms'][arm]['samples']); the label is row['moveit_arm']. Train on
scenes the diffusion model was not trained on, or its samples look better
than they will on new scenes.

Run with the LeRobot venv:
    ~/lerobot_venv/bin/python arm_classifier.py train --class-weights none \
        --data ~/friend_data/seeds_2000 --train ~/eval_friend/none.jsonl \
        --val-data ~/panda_data --val ~/eval_guide/val_none.jsonl --out ~/arm_classifier.pt
    ~/lerobot_venv/bin/python arm_classifier.py evaluate --model ~/arm_classifier.pt \
        --plans ~/eval_exec_test_v2.jsonl
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from convert_lerobot import to_base
from panda_kinematics import tcp_position

ARMS = ('first', 'second')
OTHER = {'first': 'second', 'second': 'first'}
CLASSES = ARMS + (None,)       # label index 2 = neither
ACCURATE = 0.01                # m; evaluate_policy.EXEC_TOLERANCE
MISSING_ERROR = 0.3            # m; grasp/place error when the gripper never acts
NO_OBSTACLE = 1.0              # m; obstacle gap when the scene has none
REACH = 0.855                  # m; Panda maximum reach from the base
READY = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])   # scenario.READY (SRDF 'ready')
SAMPLE_FEATURES = ('valid_fraction', 'collision_free_fraction', 'accurate_fraction', 'has_valid',
                   'best_valid_cost', 'median_cost', 'cost_std',
                   'grasp_error_min', 'grasp_error_median', 'place_error_min', 'place_error_median',
                   'best_clearance', 'median_clearance', 'median_T',
                   'cube_distance', 'target_distance')
SCENE_FEATURES = ('start_from_ready', 'cube_obstacle_gap', 'target_obstacle_gap', 'obstacles_in_reach',
                  'other_arm_to_cube', 'other_arm_to_target')
FEATURE_SETS = {'samples': SAMPLE_FEATURES, 'both': SAMPLE_FEATURES + SCENE_FEATURES}


def arm_features(arm, samples, margin, trial):
    """Summary of one arm's samples, in SAMPLE_FEATURES order (errors and distances in cm)."""
    def err(s, k):
        return min(s[k] if s[k] is not None else MISSING_ERROR, MISSING_ERROR)
    valid = [s for s in samples if s['valid'] and s['clearance'] >= margin]
    cost = np.array([s['cost'] for s in samples])
    grasp = np.array([err(s, 'grasp_error') for s in samples])
    place = np.array([err(s, 'place_error') for s in samples])
    clearance = np.array([s['clearance'] for s in samples])
    best = min(valid, key=lambda s: s['cost'], default=None)
    return [
        len(valid) / len(samples),
        float(np.mean([not s['collision'] for s in samples])),
        float(np.mean((grasp < ACCURATE) & (place < ACCURATE))),
        float(best is not None),
        best['cost'] if best else float(np.max(cost)),   # no valid sample: the worst cost, flagged by has_valid
        float(np.median(cost)),
        float(cost.std()),
        100 * grasp.min(), 100 * float(np.median(grasp)),
        100 * place.min(), 100 * float(np.median(place)),
        100 * (best['clearance'] if best else clearance.max()),
        100 * float(np.median(clearance)),
        float(np.median([s['T'] for s in samples])),
        100 * math.hypot(*to_base(arm, trial['cube_pose']['xyz'])),
        100 * math.hypot(*to_base(arm, trial['place_xyz'])),
    ]


def box_gap(xyz, box):
    """Horizontal distance from a point to an obstacle's footprint (0 inside)."""
    dx, dy = xyz[0] - box['xyz'][0], xyz[1] - box['xyz'][1]
    c, s = math.cos(box['yaw']), math.sin(box['yaw'])
    x, y = c * dx + s * dy, -s * dx + c * dy
    return math.hypot(max(abs(x) - box['size'][0] / 2, 0), max(abs(y) - box['size'][1] / 2, 0))


def scene_features(arm, scenario, trial):
    """The arm's view of the scene, in SCENE_FEATURES order (distances in cm)."""
    cube, target = trial['cube_pose']['xyz'], trial['place_xyz']
    obstacles = scenario['obstacles']
    other = tcp_position(np.asarray(scenario['arm_starts'][OTHER[arm]]), OTHER[arm])
    return [
        float(np.linalg.norm(np.asarray(scenario['arm_starts'][arm]) - READY)),
        100 * min((box_gap(cube, o) for o in obstacles), default=NO_OBSTACLE),
        100 * min((box_gap(target, o) for o in obstacles), default=NO_OBSTACLE),
        float(sum(math.hypot(*to_base(arm, o['xyz'])[:2]) < REACH for o in obstacles)),
        100 * float(np.linalg.norm(other - cube)),
        100 * float(np.linalg.norm(other - target)),
    ]


def load(paths, data, features='samples'):
    """Plan files -> features (N, 2, F), labels (N,), the rule's choice (N,), keys,
    MoveIt's best cost per arm (N, 2; nan where MoveIt found the arm infeasible)."""
    datas = [data] * len(paths) if isinstance(data, (str, Path)) or len(data) == 1 else data
    datas = [Path(d[0] if isinstance(d, list) else d).expanduser() for d in datas]
    assert len(datas) == len(paths), 'give one --data, or one per plan file'
    X, y, rule, keys, costs = [], [], [], [], []
    for path, data in zip(paths, datas):
        trials, scenarios = {}, {}
        for l in (data / 'trials.jsonl').read_text().splitlines():
            if l:
                t = json.loads(l)
                trials[(t['seed'], t['task_index'])] = t
        for l in Path(path).expanduser().read_text().splitlines():
            if not l:
                continue
            row = json.loads(l)
            trial = trials[(row['seed'], row['task_index'])]
            x = [arm_features(a, row['arms'][a]['samples'], row['clearance_margin'], trial) for a in ARMS]
            if features == 'both':
                if trial['scenario'] not in scenarios:
                    scenarios[trial['scenario']] = json.loads((data / trial['scenario']).read_text())
                x = [v + scene_features(a, scenarios[trial['scenario']], trial) for a, v in zip(ARMS, x)]
            X.append(x)
            y.append(CLASSES.index(row['moveit_arm']))
            rule.append(CLASSES.index(row['policy_arm']))
            keys.append((row['seed'], row['task_index']))
            costs.append([row['arms'][a]['moveit_cost'] if row['arms'][a]['moveit_feasible'] else np.nan
                          for a in ARMS])
    return (np.asarray(X, np.float32), np.asarray(y), np.asarray(rule), keys,
            np.asarray(costs, np.float64))


def soft_targets(y, costs, tau):
    """Training targets (N, 3). tau 0: one-hot labels. tau > 0: the feasible
    arms share the probability by softmax(-cost / tau), so a near-tie (which
    MoveIt itself decides differently from run to run) is not taught as a
    certain answer; tasks with no feasible arm stay "neither"."""
    t = np.zeros((len(y), 3), np.float32)
    if tau <= 0:
        t[np.arange(len(y)), y] = 1
        return t
    for i, (c, label) in enumerate(zip(costs, y)):
        if label == 2:
            t[i, 2] = 1
            continue
        w = np.where(np.isnan(c), 0.0, np.exp(-(np.nan_to_num(c, nan=0) - np.nanmin(c)) / tau))
        t[i, :2] = w / w.sum()
    return t


def metrics(pred, y, costs, tie=0.5):
    """Agreement with MoveIt, mean regret (MoveIt cost of the chosen arm minus
    the best, where both are arms MoveIt can do), regrets > tie, and agreement
    allowing near-ties (a feasible arm within `tie` of the best counts as right)."""
    best = np.nanmin(np.where(np.isnan(costs), np.inf, costs), axis=1)
    chosen = np.array([costs[i, p] if p < 2 else np.nan for i, p in enumerate(pred)])
    ok = (y < 2) & ~np.isnan(chosen)
    regret = chosen[ok] - best[ok]
    tolerant = np.where(y == 2, pred == 2, (pred < 2) & ~np.isnan(chosen) & (chosen - best <= tie))
    return dict(agree=float(np.mean(pred == y)), regret=float(regret.mean()), big=int(np.sum(regret > tie)),
                tolerant=float(np.mean(tolerant)))


def mlp(inputs, hidden, layers, outputs, dropout):
    mods, width = [], inputs
    for _ in range(layers):
        mods += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
        width = hidden
    return nn.Sequential(*mods, nn.Linear(width, outputs))


class SharedScorer(nn.Module):
    """Score each arm with the same MLP; the neither logit is a learned bias."""
    def __init__(self, features, hidden, layers, dropout):
        super().__init__()
        self.score = mlp(features, hidden, layers, 1, dropout)
        self.neither = nn.Parameter(torch.zeros(1))

    def forward(self, x):                                   # (B, 2, F) -> (B, 3)
        scores = self.score(x).squeeze(-1)                  # (B, 2)
        return torch.cat([scores, self.neither.expand(len(x), 1)], dim=1)


class PlainMLP(nn.Module):
    def __init__(self, features, hidden, layers, dropout):
        super().__init__()
        self.net = mlp(2 * features, hidden, layers, 3, dropout)

    def forward(self, x):
        return self.net(x.flatten(1))


MODELS = {'shared': SharedScorer, 'plain': PlainMLP}


def build(config):
    return MODELS[config['model']](len(FEATURE_SETS[config['features']]), config['hidden'], config['layers'],
                                   config['dropout'])


def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def config_of(args):
    return dict(model=args.model, features=args.features, hidden=args.hidden, layers=args.layers,
                dropout=args.dropout)


def fit(Xtr, ytr, Xva, yva, args, log=print, costs=None):
    """Train one network; with a val set keep the best epoch, else train args.epochs.
    Returns a checkpoint dict (config, state, mean, std) and the epoch kept."""
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    # Standardize with the training statistics, shared by both arms.
    mean, std = Xtr.reshape(-1, Xtr.shape[-1]).mean(0), Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    t = lambda a, dtype=torch.float32: torch.as_tensor(a, dtype=dtype, device=device)
    tau = getattr(args, 'tau', 0.0)
    ttr = t(soft_targets(ytr, costs, tau) if tau > 0 else soft_targets(ytr, None, 0))
    xtr, ytr = t((Xtr - mean) / std), t(ytr, torch.long)
    validate = Xva is not None
    if validate:
        xva, yva = t((Xva - mean) / std), t(yva, torch.long)

    # "neither" is rare (~3%): weight classes by inverse frequency (or its
    # square root, softer: full weighting makes the network say neither too often).
    counts = torch.bincount(ytr, minlength=3).float().clamp(min=1)
    weights = (counts.sum() / (3 * counts)) ** {'inverse': 1.0, 'sqrt': 0.5, 'none': 0.0}[args.class_weights]
    config = config_of(args)
    model = build(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    def train_loss(logits, target):              # soft cross-entropy, class-weighted by target
        return -(target * weights * torch.log_softmax(logits, 1)).sum(1).mean()
    log(f'{len(xtr)} train / {len(xva) if validate else 0} val trials, {xtr.shape[-1]} features per arm, '
        f'labels {counts.int().tolist()} (first, second, neither), '
        f'{sum(p.numel() for p in model.parameters())} parameters, {device}')

    best, best_state, best_epoch, since = -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for idx in torch.randperm(len(xtr), device=device).split(args.batch):
            opt.zero_grad()
            loss = train_loss(model(xtr[idx]), ttr[idx])
            loss.backward()
            opt.step()
        model.eval()
        if not validate:
            continue
        with torch.no_grad():
            val_logits = model(xva)
            val_acc, val_loss = accuracy(val_logits, yva), loss_fn(val_logits, yva).item()
        if val_acc > best:
            best, best_epoch, since = val_acc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if epoch % 25 == 0 or epoch == 1:
            with torch.no_grad():
                train_acc = accuracy(model(xtr), ytr)
            log(f'epoch {epoch}: loss {loss.item():.3f}, train {train_acc:.3f}, '
                f'val {val_acc:.3f} (loss {val_loss:.3f}, best {best:.3f})')
        if since >= args.patience:
            log(f'stopped at epoch {epoch}: no val improvement in {args.patience} epochs')
            break
    if not validate:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = args.epochs
    else:
        log(f'best val accuracy {best:.3f} at epoch {best_epoch}')
    return dict(config=config, state=best_state, mean=mean, std=std, tau=tau), best_epoch


def train(args):
    Xtr, ytr, _, _, ctr = load(args.train, args.data, args.features)
    Xva, yva, rule_va = None, None, None
    if args.val:
        Xva, yva, rule_va, _, _ = load(args.val, [args.val_data] if args.val_data else args.data, args.features)
        print(f'rule (cheapest valid sample) on val: {np.mean(rule_va == yva):.3f}')
    checkpoint, _ = fit(Xtr, ytr, Xva, yva, args, log=lambda m: print(m, flush=True), costs=ctr)
    torch.save(checkpoint, Path(args.out).expanduser())
    print(f'-> {args.out}')


def report(name, pred, y):
    return (f'{name:8s} agrees with MoveIt {np.mean(pred == y):.3f}; '
            f'neither correct {np.sum((pred == 2) & (y == 2))}/{np.sum(y == 2)}, '
            f'neither wrongly {np.sum((pred == 2) & (y != 2))}')


def crossval(args):
    """K-fold cross-validation by seed: each fold is predicted by a network
    trained on K-2 folds and early-stopped on the next fold."""
    X, y, rule, keys, costs = load(args.plans, args.data, args.features)
    seeds = np.array([k[0] for k in keys])
    unique = np.unique(seeds)
    accs, epochs, ms = [], [], []
    print(f'{len(y)} trials, {len(unique)} seeds, labels {np.bincount(y, minlength=3).tolist()}, '
          f'{args.folds} folds x {args.repeats} repeats')
    print(report('rule', rule, y))
    for repeat in range(args.repeats):
        order = np.random.default_rng(repeat).permutation(unique)
        fold = {s: i % args.folds for i, s in enumerate(order)}
        f = np.array([fold[s] for s in seeds])
        pred = np.empty_like(y)
        for k in range(args.folds):
            stop = (k + 1) % args.folds
            tr, va, te = (f != k) & (f != stop), f == stop, f == k
            args.seed = repeat * args.folds + k
            checkpoint, epoch = fit(X[tr], y[tr], X[va], y[va], args, log=lambda m: None, costs=costs[tr])
            pred[te] = predict(checkpoint, X[te], args.device).argmax(1)
            epochs.append(epoch)
        accs.append(np.mean(pred == y))
        ms.append(metrics(pred, y, costs))
        print(report(f'repeat {repeat}', pred, y) +
              f'; changed {np.sum(pred != rule)}: {np.sum((pred != rule) & (pred == y))} to the label, '
              f'{np.sum((pred != rule) & (rule == y))} away', flush=True)
    print(f'network {np.mean(accs):.3f} +- {np.std(accs):.3f} vs rule {np.mean(rule == y):.3f}; '
          f'best epoch median {int(np.median(epochs))} (use as --epochs to train on all)')
    print('network ' + fmt({k: np.mean([m[k] for m in ms]) for k in ms[0]}))
    print('rule    ' + fmt(metrics(rule, y, costs)))


def fmt(m):
    return (f'agree {100 * m["agree"]:.1f}%  tolerant {100 * m["tolerant"]:.1f}%  '
            f'regret {m["regret"]:.3f}  regret>0.5 {m["big"]:.1f}')


def curve(args):
    """Learning curve: cross-validation by seed as in crossval, with the
    training folds cut to a fraction of their seeds."""
    X, y, rule, keys, costs = load(args.plans, args.data, args.features)
    seeds = np.array([k[0] for k in keys])
    unique = np.unique(seeds)
    print(f'{len(y)} trials, {len(unique)} seeds; rule ' + fmt(metrics(rule, y, costs)), flush=True)
    for frac in args.fractions:
        ms, ntr = [], []
        for repeat in range(args.repeats):
            rng = np.random.default_rng(repeat)
            order = rng.permutation(unique)
            fold = {s: i % args.folds for i, s in enumerate(order)}
            f = np.array([fold[s] for s in seeds])
            pred = np.empty_like(y)
            for k in range(args.folds):
                stop = (k + 1) % args.folds
                pool = [s for s in order if fold[s] not in (k, stop)]
                keep = set(pool[:max(1, round(frac * len(pool)))])
                tr = np.array([s in keep for s in seeds])
                va, te = f == stop, f == k
                args.seed = repeat * args.folds + k
                checkpoint, _ = fit(X[tr], y[tr], X[va], y[va], args, log=lambda m: None, costs=costs[tr])
                pred[te] = predict(checkpoint, X[te], args.device).argmax(1)
                ntr.append(tr.sum())
            ms.append(metrics(pred, y, costs))
        print(f'fraction {frac:.2f} (~{int(np.mean(ntr))} training tasks): '
              + fmt({k: np.mean([m[k] for m in ms]) for k in ms[0]}), flush=True)


def predict(checkpoint, X, device):
    """Class probabilities (N, 3) for features (N, 2, F). A list of
    checkpoints is an ensemble: their probabilities are averaged."""
    if isinstance(checkpoint, list):
        return np.mean([predict(c, X, device) for c in checkpoint], axis=0)
    model = build(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['state'])
    model.eval()
    x = torch.as_tensor((X - checkpoint['mean']) / checkpoint['std'], dtype=torch.float32, device=device)
    with torch.no_grad():
        return torch.softmax(model(x), dim=1).cpu().numpy()


def load_models(paths):
    checkpoints = [torch.load(Path(p).expanduser(), weights_only=False) for p in paths]
    assert len({c['config']['features'] for c in checkpoints}) == 1, 'ensemble members need the same features'
    return checkpoints


def evaluate(args):
    checkpoint = load_models(args.model)
    X, y, rule, _, costs = load(args.plans, args.data, checkpoint[0]['config']['features'])
    probs = predict(checkpoint, X, args.device)
    if args.usable_only:
        probs[:, :2] *= X[:, :, SAMPLE_FEATURES.index('has_valid')]
    pred = probs.argmax(1)
    print(f'{len(y)} trials; labels: {np.bincount(y, minlength=3).tolist()} (first, second, neither)')
    print(report('rule', rule, y))
    print(report('network', pred, y))
    print('rule    ' + fmt(metrics(rule, y, costs)))
    print('network ' + fmt(metrics(pred, y, costs)))
    if args.save_pred:
        np.save(Path(args.save_pred).expanduser(), np.stack([pred, rule, y]))
    print(f'network changes the rule\'s choice in {np.sum(pred != rule)} trials: '
          f'{np.sum((pred != rule) & (pred == y))} to the label, {np.sum((pred != rule) & (rule == y))} away from it')


def relabel(args):
    """Write --plans with the network's arm in policy_arm, ready for
    execute_policy.py: the candidates are reordered so the chosen arm's come
    first. Only arms with a usable sample can be chosen (there is nothing to
    execute otherwise), so the network picks among those and neither."""
    checkpoint = load_models(args.model)
    X, y, rule, keys, _ = load(args.plans, args.data, checkpoint[0]['config']['features'])
    probs = predict(checkpoint, X, args.device)
    probs[:, :2] *= X[:, :, SAMPLE_FEATURES.index('has_valid')]
    pred = probs.argmax(1)
    print(report('rule', rule, y))
    print(report('network', pred, y))
    rows = [json.loads(l) for path in args.plans for l in Path(path).expanduser().read_text().splitlines() if l]
    assert [(r['seed'], r['task_index']) for r in rows] == keys
    changed = without = 0
    with Path(args.out).expanduser().open('w') as out:
        for row, c in zip(rows, pred):
            arm = CLASSES[c]
            row['rule_arm'], row['nn_arm'] = row['policy_arm'], arm
            if arm != row['policy_arm']:
                changed += 1
                candidates = row.get('candidates') or []
                mine = [x for x in candidates if x['arm'] == arm]
                if arm and not mine:
                    without += 1
                    arm = None
                row['policy_arm'] = arm
                if arm:
                    row['candidates'] = mine + [x for x in candidates if x['arm'] != arm]
                    row['trajectory'] = row['candidates'][0]['trajectory']
                    row['trajectory_score'] = row['candidates'][0]['score']
                else:
                    for k in ('candidates', 'trajectory', 'trajectory_score'):
                        row.pop(k, None)
            out.write(json.dumps(row) + '\n')
    print(f'{len(rows)} trials, network changed {changed} '
          f'({without} to an arm with no usable sample: declined) -> {args.out}')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('train', 'crossval', 'curve', 'evaluate', 'relabel'):
        s = sub.add_parser(name)
        s.add_argument('--data', nargs='+', default=[str(Path.home() / 'panda_data')],
                       help='dataset with trials.jsonl (one, or one per plan file)')
        s.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    for name in ('train', 'crossval', 'curve'):
        t = sub.choices[name]
        t.add_argument('--model', choices=MODELS, default='shared')
        t.add_argument('--features', choices=FEATURE_SETS, default='samples',
                       help='samples: sample summaries only; both: plus scene features')
        t.add_argument('--hidden', type=int, default=64)
        t.add_argument('--layers', type=int, default=2, help='hidden layers')
        t.add_argument('--dropout', type=float, default=0.1)
        t.add_argument('--lr', type=float, default=1e-3)
        t.add_argument('--weight-decay', type=float, default=1e-4)
        t.add_argument('--class-weights', choices=('inverse', 'sqrt', 'none'), default='inverse',
                       help='loss weight per class from its frequency')
        t.add_argument('--batch', type=int, default=128)
        t.add_argument('--epochs', type=int, default=300)
        t.add_argument('--patience', type=int, default=50, help='epochs without val improvement before stopping')
        t.add_argument('--tau', type=float, default=0.0,
                       help='soft targets from MoveIt costs, softmax(-cost / tau) over feasible arms (0: one-hot)')
    t = sub.choices['train']
    t.add_argument('--train', nargs='+', required=True, help='plan files (evaluate_policy.py --out)')
    t.add_argument('--val', nargs='+', help='plan files for early stopping (without: train --epochs)')
    t.add_argument('--val-data', help='dataset of the --val plans (default: --data)')
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--out', default=str(Path.home() / 'arm_classifier.pt'))
    for name in ('crossval', 'curve'):
        c = sub.choices[name]
        c.add_argument('--plans', nargs='+', required=True)
        c.add_argument('--folds', type=int, default=5)
        c.add_argument('--repeats', type=int, default=3, help='fold splits and initializations')
    sub.choices['curve'].add_argument('--fractions', type=float, nargs='+', default=[0.15, 0.25, 0.5, 0.75, 1.0])
    e = sub.choices['evaluate']
    e.add_argument('--model', nargs='+', required=True, help='checkpoint(s) from train; several = ensemble')
    e.add_argument('--plans', nargs='+', required=True)
    e.add_argument('--usable-only', action='store_true', help='choose only arms with a usable sample, as relabel')
    e.add_argument('--save-pred', help='save (prediction, rule, label) as .npy')
    r = sub.choices['relabel']
    r.add_argument('--model', nargs='+', required=True, help='checkpoint(s) from train; several = ensemble')
    r.add_argument('--plans', nargs='+', required=True, help='evaluate_policy.py --save-trajectories output')
    r.add_argument('--out', required=True)
    args = p.parse_args()
    dict(train=train, crossval=crossval, curve=curve, evaluate=evaluate, relabel=relabel)[args.command](args)


if __name__ == '__main__':
    main()
