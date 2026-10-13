"""Fee-exact portfolio rebalancing shared by baselines, environments and the benchmark."""

import numpy as np


def rebalance_positions(cash, positions, weights, transaction_cost_pct):
    """Solve self-financing fractional targets at the current execution prices.

    ``positions`` holds the pre-trade value of each asset at the execution
    prices and ``weights`` the targets as fractions of post-fee equity.
    Returns ``(target_positions, cash, cost)``.

    For pre-trade equity B and post-fee equity V, solve
    V + fee * sum(abs(weights * V - positions)) = B. The left side is
    piecewise linear and convex. Every non-final Newton step from B crosses
    a breakpoint, so the solver needs at most one step per position plus
    the final linear solve. A reusable work array avoids per-step allocations.
    """
    fee = float(transaction_cost_pct)
    if not np.isfinite(fee) or fee < 0:
        raise ValueError("transaction_cost_pct must be finite and nonnegative")
    total = float(cash + positions.sum())
    net = total
    cost = 0.0
    if fee:
        work = np.empty_like(positions)
        for _ in range(positions.size + 2):
            np.multiply(weights, net, out=work)
            np.subtract(work, positions, out=work)
            np.sign(work, out=work)
            slope = 1 + fee * np.dot(work, weights)
            if slope <= 0:
                raise ValueError("Transaction costs leave no self-financing portfolio")
            adjusted = (total + fee * np.dot(work, positions)) / slope
            if adjusted < 0:
                raise ValueError("Transaction costs leave no self-financing portfolio")
            converged = abs(adjusted - net) <= 4 * np.spacing(max(1.0, abs(total)))
            net = adjusted
            if converged:
                break
        else:
            raise ArithmeticError("Fee-adjusted portfolio rebalance did not converge")
    target = weights * net
    if fee:
        np.subtract(target, positions, out=work)
        np.abs(work, out=work)
        cost = float(fee * work.sum())
    return target, float(total - target.sum() - cost), cost
