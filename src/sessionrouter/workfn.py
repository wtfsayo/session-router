"""Work Function Algorithm — the metrical-task-system core.

States = models. Per-turn task = vector of per-model hitting costs
(price + quality deficit + latency). Movement cost d(i,j) = re-prefill on j.

WFA (Borodin–Linial–Saks): w_t(i) = min_j { w_{t-1}(j) + c_t(j) + d(j,i) };
move to argmin_i { w_t(i) + d(s, i) }. Deterministic 2n-1 competitive —
optimal. For n=2 it reduces to a break-even (ski-rental) rule with correct
accounting in both directions.

Switch costs here are destination-determined (the *receiving* model re-prefills),
so we symmetrize: d'(i,j) = (d(i,j) + d(j,i))/2. Total cost under d' differs
from the true cost by <= (w_first + w_last)/2 — an additive constant.
"""
from __future__ import annotations


class WorkFunction:
    def __init__(self, n: int):
        self.n = n
        self.w = [0.0] * n
        self.state = 0

    def step(self, costs: list[float], d: list[list[float]]) -> int:
        """costs[i] = hitting cost on model i this turn; d[i][j] = switch i->j."""
        n = self.n
        # symmetrize
        ds = [[(d[i][j] + d[j][i]) / 2 for j in range(n)] for i in range(n)]
        w_new = [0.0] * n
        for i in range(n):
            w_new[i] = min(self.w[j] + costs[j] + ds[j][i] for j in range(n))
        # Move rule: argmin w(i) + d(state, i). In 2-state constant-cost
        # regimes the work-function gap saturates at the metric distance and
        # this expression ties forever — break ties toward min w(i) (the
        # state OPT prefers), giving break-even switching behavior.
        moves = [w_new[i] + ds[self.state][i] for i in range(n)]
        best = min(moves)
        tied = [i for i in range(n) if moves[i] <= best + 1e-12]
        s_new = min(tied, key=lambda i: w_new[i])
        self.w, self.state = w_new, s_new
        return s_new


def hitting_cost(price_per_mtok: float, tokens: int,
                 quality_deficit: float, quality_weight: float,
                 est_output_tokens: int = 0, output_price: float = 0.0) -> float:
    """Per-turn cost on a model, in dollars.
    quality_deficit in [0,1] monetized at quality_weight $/unit."""
    return (tokens * price_per_mtok + est_output_tokens * output_price) / 1e6 \
        + quality_weight * quality_deficit
