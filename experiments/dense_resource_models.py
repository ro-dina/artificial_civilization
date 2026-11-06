"""Fixed, deterministic additive categorical logistic regression. Analysis only."""
import json
import numpy as np

from experiments.nonstationary_prediction import Hierarchy
from experiments.future_resource_prediction import keys as h_keys

FIELDS = [(f'need{i}', list(range(3))) for i in range(3)] + [(f'resource{i}', [-1, 0, 1, 2]) for i in range(4)] + [
    ('auditory_kind', ['SILENCE', 'MASKED', 'IDENTIFIED']), ('signal', [None, *range(16)]),
    ('direction', [None, 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'SAME_CELL']),
    ('auditory_age', [None, 0, 1, 2, 3]), ('action', list(range(8))), ('B', list(range(11))), ('RD', list(range(6)))]


class Encoding:
    def __init__(self, level):
        if level not in (0, 1, 2):
            raise ValueError('Nested encoding level0/1/2 required')
        self.fields = FIELDS[:12 + level]
        offset = 1
        self.maps = []
        for name, values in self.fields:
            self.maps.append({v: offset + i for i, v in enumerate(values)})
            offset += len(values)
        self.dimension = offset

    @staticmethod
    def original(key):
        x = json.loads(key)
        if len(x) != 8:
            raise ValueError('Expected original7 body/resource fields + auditory tuple')
        auditory = x[7]
        if auditory[0] == 'SILENCE' and len(auditory) == 1:
            sound = ['SILENCE', None, None, None]
        elif auditory[0] == 'MASKED' and len(auditory) == 2:
            sound = ['MASKED', None, None, auditory[1]]
        elif auditory[0] == 'IDENTIFIED' and len(auditory) == 4:
            sound = [auditory[0], auditory[1], auditory[2], auditory[3]]
        else:
            raise ValueError('Unsupported auditory feature')
        return [*x[:7], *sound]

    def transform(self, frame):
        lookup = {key: self.original(key) for key in frame.x_key.unique()}
        base = np.array([lookup[k] for k in frame.x_key], dtype=object).reshape(len(frame), 11)
        extra = [frame.action.to_numpy(), frame.b_token.to_numpy(), frame.rd_token.to_numpy()]
        values = [base[:, i] for i in range(11)] + extra[:len(self.fields) - 11]
        result = np.zeros((len(frame), len(self.fields) + 1), dtype=np.int16)
        try:
            for j, (mapping, column) in enumerate(zip(self.maps, values), 1):
                result[:, j] = [mapping[v] for v in column]
        except KeyError as error:
            raise ValueError('Origin feature outside frozen categories: ' + str(error)) from error
        return result


def sigmoid(value):
    positive = value >= 0
    result = np.empty_like(value, dtype=float)
    result[positive] = 1 / (1 + np.exp(-value[positive]))
    ex = np.exp(value[~positive]); result[~positive] = ex / (1 + ex)
    return result


def objective_gradient_hessian(coefficients, indices, positive, count, l2=.001, hessian=True):
    n, d = count.sum(), len(coefficients)
    logits = coefficients[indices].sum(axis=1)
    probability = sigmoid(logits)
    objective = (count * np.logaddexp(0, logits) - positive * logits).sum() / n + .5 * l2 * np.dot(coefficients[1:], coefficients[1:])
    residual = (count * probability - positive) / n
    gradient = np.bincount(indices.ravel(), weights=np.repeat(residual, indices.shape[1]), minlength=d)
    gradient[1:] += l2 * coefficients[1:]
    matrix = None
    if hessian:
        weight = count * probability * (1 - probability) / n
        matrix = np.diag(np.r_[0., np.full(d - 1, l2)])
        for j in range(indices.shape[1]):
            for k in range(j, indices.shape[1]):
                block = np.bincount(indices[:, j].astype(np.int64) * d + indices[:, k], weights=weight, minlength=d * d).reshape(d, d)
                matrix += block if j == k else block + block.T
    return float(objective), gradient, matrix


def fit_logistic(indices, positive, count, dimension, l2=.001, tolerance=1e-8, iterations=100):
    positive, count = np.asarray(positive, float), np.asarray(count, float)
    if len(indices) != len(count) or (count <= 0).any() or (positive < 0).any() or (positive > count).any() or not 0 < positive.sum() < count.sum():
        raise ValueError('Positive weights and both observed classes required')
    # Exact sufficient-statistic collapse; no sampling/subset or RNG.
    unique, inverse = np.unique(indices, axis=0, return_inverse=True)
    yes = np.bincount(inverse, weights=positive); weights = np.bincount(inverse, weights=count)
    coefficients = np.zeros(dimension)
    prevalence = yes.sum() / weights.sum(); coefficients[0] = np.log(prevalence / (1 - prevalence))
    converged = False
    for iteration in range(iterations + 1):
        objective, gradient, matrix = objective_gradient_hessian(coefficients, unique, yes, weights, l2)
        norm = float(np.max(np.abs(gradient)))
        if norm <= tolerance:
            converged = True; break
        if iteration == iterations:
            break
        delta = np.linalg.solve(matrix, gradient)
        rate = 1.
        for _ in range(31):
            candidate = coefficients - rate * delta
            next_objective, _, _ = objective_gradient_hessian(candidate, unique, yes, weights, l2, False)
            if next_objective <= objective - 1e-4 * rate * np.dot(gradient, delta):
                coefficients = candidate; break
            rate *= .5
        else:
            break
    return {'coefficients': coefficients.tolist(), 'dimension': dimension, 'converged': converged, 'iterations': iteration,
            'gradient_infinity': norm, 'objective': objective, 'unique_patterns': len(unique), 'training_decisions': int(weights.sum()),
            'positive': int(yes.sum()), 'l2': l2, 'tolerance': tolerance}


def hierarchy_from_patterns(frame):
    result = Hierarchy(('X_A', 'X_B_A', 'X_C_A'), 2, minimum=20, strength=5)
    for row in frame.itertuples(index=False):
        values = [int(row.count - row.positive), int(row.positive)]
        action = int(row.action)
        prior = result.actions.setdefault(action, [0, 0])
        for i in (0, 1): prior[i] += values[i]
        for head, key in h_keys(row.x_key, action, int(row.b_token), int(row.rd_token)).items():
            cell = result.cells[head].setdefault(key, [0, 0])
            for i in (0, 1): cell[i] += values[i]
    result.freeze()
    return result


def feature_counts(indices, dimension, weights=None):
    return np.bincount(indices.ravel(), weights=np.repeat(weights, indices.shape[1]) if weights is not None else None, minlength=dimension)


def feature_support(indices, counts, minimum=20):
    # Intercept excluded from category-level qualification.
    return (counts[indices[:, 1:]] >= minimum).all(axis=1)
