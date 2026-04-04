"""Domain scale presets per implementation spec §5."""

from earned_understanding_battery.core.types import DomainConfig

SMALL = DomainConfig(
    n_nodes=50,
    n_communities=4,
    p_within=0.3,
    p_between=0.03,
    seed=42,
)

MEDIUM = DomainConfig(
    n_nodes=150,
    n_communities=6,
    p_within=0.3,
    p_between=0.02,
    seed=42,
)

LARGE = DomainConfig(
    n_nodes=500,
    n_communities=12,
    p_within=0.25,
    p_between=0.015,
    seed=42,
)

ALL_PRESETS = {
    "small": SMALL,
    "medium": MEDIUM,
    "large": LARGE,
}
