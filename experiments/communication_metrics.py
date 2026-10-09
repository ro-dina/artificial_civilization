"""Discrete, base-2 empirical measures and reproducible analysis-only nulls.

These are descriptive estimates, not tests for language, causality or significance.
"""

import hashlib

import numpy as np


def analysis_rng(seed: int, *labels) -> np.random.Generator:
    payload = repr((seed, *labels)).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:16], "big"))


def codes(values) -> np.ndarray:
    mapping = {}
    return np.asarray([mapping.setdefault(value, len(mapping)) for value in values], dtype=np.int64)


def entropy(counts) -> float | None:
    values = np.asarray(list(counts), dtype=float)
    if (values < 0).any() or not np.isfinite(values).all():
        raise ValueError("Entropy requires finite, nonnegative frequencies")
    total = values.sum()
    if total == 0:
        return None
    probabilities = values[values > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def _entropy_rows(counts):
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=1)
    p = np.divide(counts, totals[:, None], out=np.zeros_like(counts), where=totals[:, None] > 0)
    logs = np.zeros_like(p)
    np.log2(p, out=logs, where=p > 0)
    result = -(p * logs).sum(axis=1)
    result[totals == 0] = np.nan
    return result


def grouped_information(v, x, groups):
    """Per-stratum MI and entropies; no pool of agents or temporal independence claim."""
    if not (len(v) == len(x) == len(groups)) or not len(v):
        raise ValueError("Grouped information requires equally sized nonempty arrays")
    nv, nx, ng = int(max(v)) + 1, int(max(x)) + 1, int(max(groups)) + 1
    counts = np.bincount((groups * nv + v) * nx + x, minlength=ng * nv * nx).reshape(ng, nv, nx)
    n = counts.sum(axis=(1, 2))
    hv = _entropy_rows(counts.sum(axis=2))
    hx = _entropy_rows(counts.sum(axis=1))
    h_joint = _entropy_rows(counts.reshape(ng, -1))
    return np.maximum(0, hv + hx - h_joint), hv, hx, n


def mutual_information(x, y) -> float | None:
    if len(x) != len(y):
        raise ValueError("MI inputs must have equal lengths")
    if not len(x):
        return None
    result, _, _, _ = grouped_information(codes(x), codes(y), np.zeros(len(x), dtype=np.int64))
    return float(result[0])


def information_report(v, x, *, permutations: int, rng, groups=None):
    """Shuffle V globally, or within fixed strata; retain counts/structure exactly.

    Returns a weighted conditional summary and individual/stratum diagnostics.
    Corrections may be negative. SD is sample SD across permutations, ddof=1.
    """
    if type(permutations) is not int or permutations < 1:
        raise ValueError("permutations must be a positive integer")
    if len(v) != len(x):
        raise ValueError("MI inputs must have equal lengths")
    if not len(v):
        return {name: None for name in ("mi_bits", "h_v_bits", "h_x_bits", "nmi", "permuted_mean_bits",
                                        "permuted_std_bits", "corrected_mi_bits")} | {"n": 0, "strata": 0}, []
    v, x = codes(v), codes(x)
    g = np.zeros(len(v), dtype=np.int64) if groups is None else codes(groups)
    if len(g) != len(v):
        raise ValueError("Permutation groups must match input lengths")
    observed, hv, hx, n = grouped_information(v, x, g)
    weights = n / n.sum()
    null = np.zeros((permutations, len(n)))
    # Sorting/group boundaries once avoids an O(N * agents) scan per shuffle.
    order = np.argsort(g, kind="stable")
    segments = np.split(order, np.flatnonzero(np.diff(g[order])) + 1)
    # Equal-length strata form matrix rows. Generator.permuted independently
    # shuffles each row, preserving the exact same within-stratum null model
    # without a Python RNG call for every individual and permutation.
    buckets = {}
    for index, indices in enumerate(segments):
        if hv[index] > 0 and hx[index] > 0:
            buckets.setdefault(len(indices), []).append(indices)
    matrices = [np.asarray(buckets[size]) for size in sorted(buckets)]
    if matrices:
        for iteration in range(permutations):
            permuted = v.copy()
            for indices in matrices:
                permuted[indices] = rng.permuted(v[indices], axis=1)
            null[iteration], _, _, _ = grouped_information(permuted, x, g)
    pooled_null = null @ weights
    def row(mi, a, b, samples, baseline):
        denominator = min(a, b)
        mean = float(baseline.mean())
        return {"n": int(samples), "mi_bits": float(mi), "h_v_bits": float(a), "h_x_bits": float(b),
                "nmi": float(mi / denominator) if denominator > 0 else None,
                "permuted_mean_bits": mean,
                "permuted_std_bits": float(baseline.std(ddof=1)) if permutations > 1 else None,
                "corrected_mi_bits": float(mi - mean)}
    summary = row(observed @ weights, hv @ weights, hx @ weights, n.sum(), pooled_null)
    summary["strata"] = len(n)
    individual = [row(observed[i], hv[i], hx[i], n[i], null[:, i]) for i in range(len(n))]
    return summary, individual


def jensen_shannon(p, q) -> float | None:
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    if p.shape != q.shape or p.ndim != 1 or (p < 0).any() or (q < 0).any() or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("JSD requires equal, finite, nonnegative frequency vectors")
    if p.sum() == 0 or q.sum() == 0:
        return None
    p, q = p / p.sum(), q / q.sum()
    return float(np.clip(entropy((p + q) / 2) - (entropy(p) + entropy(q)) / 2, 0, 1))


def pair_sample(agents: int, limit: int, rng):
    total = agents * (agents - 1) // 2
    ranks = np.arange(total) if total <= limit else np.sort(rng.choice(total, limit, replace=False))
    pairs = []
    # Unrank uniformly sampled triangular positions without an all-pairs list.
    for rank in ranks:
        rank = int(rank)
        lo, hi = 0, agents - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            before = mid * (2 * agents - mid - 1) // 2
            if before <= rank:
                lo = mid
            else:
                hi = mid
        before = lo * (2 * agents - lo - 1) // 2
        pairs.append((lo, lo + 1 + rank - before))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2), total


def pair_jsd(counts, pairs):
    probabilities = counts / counts.sum(axis=1)[:, None]
    p, q = probabilities[pairs[:, 0]], probabilities[pairs[:, 1]]
    return np.clip(_entropy_rows((p + q) / 2) - (_entropy_rows(p) + _entropy_rows(q)) / 2, 0, 1)


def convergence(signals, agent_ids, *, vocabulary, min_samples, max_pairs, permutations, rng):
    sizes = {}
    for agent in agent_ids:
        sizes[agent] = sizes.get(agent, 0) + 1
    eligible = sorted(agent for agent, n in sizes.items() if n >= min_samples)
    mapping = {agent: index for index, agent in enumerate(eligible)}
    selected = [(signal, mapping[agent]) for signal, agent in zip(signals, agent_ids) if agent in mapping]
    pairs, possible = pair_sample(len(eligible), max_pairs, rng)
    summary = {"eligible_agents": len(eligible), "pairs": len(pairs), "possible_pairs": possible,
               "n_emitted": len(selected), "pair_method": "exhaustive" if possible <= max_pairs else "uniform seeded ranks",
               "min_emitted_samples": min_samples, "mean_jsd_bits": None, "median_jsd_bits": None,
               "null_mean_jsd_bits": None, "null_std_jsd_bits": None, "observed_minus_null_bits": None}
    if not len(pairs):
        return summary
    signals = np.asarray([s for s, _ in selected], dtype=np.int64)
    groups = np.asarray([g for _, g in selected], dtype=np.int64)
    counts = np.bincount(groups * vocabulary + signals, minlength=len(eligible) * vocabulary).reshape(len(eligible), vocabulary)
    values = pair_jsd(counts, pairs)
    baseline = []
    for _ in range(permutations):
        permuted = rng.permutation(signals)
        null_counts = np.bincount(groups * vocabulary + permuted, minlength=counts.size).reshape(counts.shape)
        baseline.append(float(pair_jsd(null_counts, pairs).mean()))
    summary.update(mean_jsd_bits=float(values.mean()), median_jsd_bits=float(np.median(values)),
                   null_mean_jsd_bits=float(np.mean(baseline)),
                   null_std_jsd_bits=float(np.std(baseline, ddof=1)) if permutations > 1 else None,
                   observed_minus_null_bits=float(values.mean() - np.mean(baseline)))
    return summary
