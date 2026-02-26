"""
it2-evidence-framework
======================
Bounded uncertainty propagation in normalised interval Type-2 fuzzy aggregation.

Ramesh & Mehreen (2025)
"""
from .membership        import sigmoid, interval_membership, interval_width
from .aggregation       import normalised_aggregate, it2_aggregate_interval
from .corner_evaluation import corner_evaluation, corner_evaluation_batch
from .width_bounds      import width_bound_lipschitz, certified_width_o1
from .yager             import yager_aggregate, yager_width_bound
from .concentration     import bernstein_bound, hoeffding_bound

__all__ = [
    "sigmoid", "interval_membership", "interval_width",
    "normalised_aggregate", "it2_aggregate_interval",
    "corner_evaluation", "corner_evaluation_batch",
    "width_bound_lipschitz", "certified_width_o1",
    "yager_aggregate", "yager_width_bound",
    "bernstein_bound", "hoeffding_bound",
]
