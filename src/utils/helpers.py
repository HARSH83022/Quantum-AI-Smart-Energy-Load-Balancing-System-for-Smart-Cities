"""
Utility helpers for the Quantum AI Smart Grid system.

Provides shared transformer configuration, time conversion,
and load balance scoring utilities used across modules.
"""

import numpy as np


# ─────────────────────────────────────────────
# Delhi Grid Transformer Configuration
# ─────────────────────────────────────────────

def get_transformer_config():
    """
    Returns the Delhi power grid transformer configuration.

    Each transformer (zone) has:
        - name:        Zone identifier
        - capacity_mw: Rated maximum capacity in MW
        - priority:    Load allocation priority (1 = highest)

    The capacities are based on Delhi's regional load distribution
    pattern across 5 major grid zones.
    """
    return [
        {"name": "North Delhi",   "capacity_mw": 1500.0, "priority": 1},
        {"name": "South Delhi",   "capacity_mw": 1800.0, "priority": 2},
        {"name": "East Delhi",    "capacity_mw": 1200.0, "priority": 3},
        {"name": "West Delhi",    "capacity_mw": 1400.0, "priority": 4},
        {"name": "Central Delhi", "capacity_mw": 1100.0, "priority": 5},
    ]


def get_total_grid_capacity():
    """Returns the total rated capacity across all transformers (MW)."""
    return sum(t["capacity_mw"] for t in get_transformer_config())


# ─────────────────────────────────────────────
# Time Conversion
# ─────────────────────────────────────────────

def decimal_hours_to_time_str(hours):
    """
    Convert decimal hours to HH:MM format string.

    Examples:
        14.5  → "14:30"
        0.75  → "00:45"
        23.99 → "23:59"
    """
    h = int(hours)
    m = int(round((hours - h) * 60))
    if m == 60:
        h += 1
        m = 0
    return f"{h:02d}:{m:02d}"


# ─────────────────────────────────────────────
# Load Balance Scoring
# ─────────────────────────────────────────────

def compute_balance_score(allocations, capacities):
    """
    Compute a load balance score between 0 and 1.

    The score measures how evenly the load is distributed
    relative to each transformer's capacity. A score of 1.0
    means all transformers are at the same utilization ratio.

    Parameters:
        allocations (list): Assigned load per transformer (MW)
        capacities  (list): Rated capacity per transformer (MW)

    Returns:
        float: Balance score in [0, 1]
    """
    allocations = np.array(allocations, dtype=float)
    capacities = np.array(capacities, dtype=float)

    # Utilization ratios
    utilizations = allocations / np.maximum(capacities, 1e-9)

    # Perfect balance → all utilizations are equal
    # Score = 1 - normalized std deviation of utilizations
    if np.max(utilizations) < 1e-9:
        return 1.0

    std_dev = np.std(utilizations)
    mean_util = np.mean(utilizations)

    # Coefficient of variation (normalized)
    cv = std_dev / max(mean_util, 1e-9)

    # Map CV to [0, 1] score (CV of 0 = perfect, CV > 1 = terrible)
    score = max(0.0, 1.0 - cv)

    return round(score, 4)


def get_transformer_status(utilization_pct):
    """
    Determine transformer status based on utilization percentage.

    Returns:
        str: Status label with emoji indicator
    """
    if utilization_pct > 95:
        return "Critical 🔴"
    elif utilization_pct > 85:
        return "Overloaded ⚠️"
    elif utilization_pct > 70:
        return "High Load 🟠"
    elif utilization_pct > 40:
        return "Optimal 🟢"
    else:
        return "Underutilized 🟡"
