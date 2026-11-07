from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

Array = np.ndarray
Tensorish = torch.Tensor | Array


def _to_numpy(value: Tensorish) -> Array:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def top_k_accuracy(probs: Tensorish, labels: Tensorish, k: int = 5) -> float:
    p = _to_numpy(probs)
    y = _to_numpy(labels).astype(np.int64)
    top = np.argsort(-p, axis=1)[:, :k]
    hits = np.any(top == y[:, None], axis=1)
    return float(hits.mean())


def _binary_auc(y: Array, score: Array) -> float | None:
    positives = score[y == 1]
    negatives = score[y == 0]
    if positives.size == 0 or negatives.size == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    unique, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    tie_sum = np.cumsum(counts)
    tie_start = tie_sum - counts + 1
    avg_rank = (tie_start + tie_sum) / 2.0
    ranks = avg_rank[inverse]
    rank_sum = ranks[y == 1].sum()
    n_pos = float(positives.size)
    n_neg = float(negatives.size)
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc_ovr(probs: Tensorish, labels: Tensorish) -> float:
    p = _to_numpy(probs)
    y = _to_numpy(labels).astype(np.int64)
    scores: list[float] = []
    for cls in range(p.shape[1]):
        auc = _binary_auc((y == cls).astype(np.int64), p[:, cls])
        if auc is not None:
            scores.append(auc)
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _per_class_counts(preds: Array, labels: Array, n_classes: int) -> tuple[Array, Array, Array]:
    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)
    for cls in range(n_classes):
        tp[cls] = np.sum((preds == cls) & (labels == cls))
        fp[cls] = np.sum((preds == cls) & (labels != cls))
        fn[cls] = np.sum((preds != cls) & (labels == cls))
    return tp, fp, fn


def macro_f1(probs: Tensorish, labels: Tensorish) -> float:
    p = _to_numpy(probs)
    y = _to_numpy(labels).astype(np.int64)
    preds = np.argmax(p, axis=1)
    tp, fp, fn = _per_class_counts(preds, y, p.shape[1])
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(f1.mean())


def macro_sensitivity(probs: Tensorish, labels: Tensorish) -> float:
    p = _to_numpy(probs)
    y = _to_numpy(labels).astype(np.int64)
    preds = np.argmax(p, axis=1)
    tp, _, fn = _per_class_counts(preds, y, p.shape[1])
    support = tp + fn
    recall = np.divide(tp, support, out=np.zeros_like(support), where=support > 0)
    present = support > 0
    return float(recall[present].mean()) if present.any() else float("nan")


def coverage(sets: Sequence[Sequence[int]], labels: Tensorish) -> float:
    y = _to_numpy(labels).astype(np.int64)
    hits = [1.0 if int(y[i]) in set(sets[i]) else 0.0 for i in range(len(sets))]
    return float(np.mean(hits))


def mean_set_size(sets: Sequence[Sequence[int]]) -> float:
    return float(np.mean([len(s) for s in sets]))


def idi(p_old: Tensorish, p_new: Tensorish, labels: Tensorish) -> float:
    a = _to_numpy(p_old)
    b = _to_numpy(p_new)
    y = _to_numpy(labels).astype(np.int64)
    event = y == 1
    nonevent = y == 0
    disc_new = b[event].mean() - b[nonevent].mean()
    disc_old = a[event].mean() - a[nonevent].mean()
    return float(disc_new - disc_old)


def nri(p_old: Tensorish, p_new: Tensorish, labels: Tensorish) -> float:
    a = _to_numpy(p_old)
    b = _to_numpy(p_new)
    y = _to_numpy(labels).astype(np.int64)
    event = y == 1
    nonevent = y == 0
    up = b > a
    down = b < a
    event_term = up[event].mean() - down[event].mean()
    nonevent_term = down[nonevent].mean() - up[nonevent].mean()
    return float(event_term + nonevent_term)


def net_benefit(prob: Tensorish, labels: Tensorish, threshold: float) -> float:
    p = _to_numpy(prob)
    y = _to_numpy(labels).astype(np.int64)
    n = float(p.size)
    flagged = p >= threshold
    tp = float(np.sum(flagged & (y == 1)))
    fp = float(np.sum(flagged & (y == 0)))
    odds = threshold / (1.0 - threshold)
    return float(tp / n - (fp / n) * odds)


def brier_score(prob: Tensorish, labels: Tensorish) -> float:
    p = _to_numpy(prob)
    y = _to_numpy(labels).astype(np.float64)
    return float(np.mean((p - y) ** 2))


def calibration_slope(prob: Tensorish, labels: Tensorish) -> float:
    p = np.clip(_to_numpy(prob).astype(np.float64), 1e-6, 1.0 - 1e-6)
    y = _to_numpy(labels).astype(np.float64)
    logit = np.log(p / (1.0 - p))
    design = np.stack([np.ones_like(logit), logit], axis=1)
    beta = np.zeros(2)
    for _ in range(25):
        eta = design @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        weight = np.clip(mu * (1.0 - mu), 1e-6, None)
        gradient = design.T @ (mu - y)
        hessian = design.T @ (design * weight[:, None]) + 1e-6 * np.eye(2)
        beta = beta - np.linalg.solve(hessian, gradient)
    return float(beta[1])


def benjamini_hochberg(pvalues: Sequence[float], q: float = 0.05) -> Array:
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    thresholds = q * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresholds
    rejected = np.zeros(m, dtype=bool)
    if passed.any():
        cutoff = np.max(np.nonzero(passed)[0])
        rejected[order[: cutoff + 1]] = True
    return rejected


def bootstrap_ci(
    metric: Callable[..., float],
    arrays: Sequence[Tensorish],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    materialised = [_to_numpy(a) for a in arrays]
    n = materialised[0].shape[0]
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        estimates[b] = metric(*[a[idx] for a in materialised])
    lo = float(np.quantile(estimates, alpha / 2.0))
    hi = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return lo, hi


def subgroup_fairness(
    metric: Callable[[Array, Array], float],
    probs: Tensorish,
    labels: Tensorish,
    groups: Tensorish,
) -> dict[int, float]:
    p = _to_numpy(probs)
    y = _to_numpy(labels)
    g = _to_numpy(groups).astype(np.int64)
    report: dict[int, float] = {}
    for value in np.unique(g):
        mask = g == value
        report[int(value)] = metric(p[mask], y[mask])
    return report
