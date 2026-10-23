"""Gera uma base fictícia e reproduzível para demonstrar o MVP de CRM."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FILE = PROJECT_ROOT / "data" / "gold" / "passageiros_sinteticos.csv"
SEED = 20261023


def generate_passengers(size: int = 1_000, seed: int = SEED) -> pd.DataFrame:
    """Cria registros artificiais; nenhum campo corresponde a um passageiro real."""
    if size < 1:
        raise ValueError("size deve ser maior que zero.")

    rng = np.random.default_rng(seed)
    flights = rng.poisson(5, size) + 1
    flights_previous = np.maximum(0, flights + rng.integers(-4, 5, size))
    months_since_last_flight = rng.integers(0, 13, size)
    avg_ticket_brl = np.round(rng.lognormal(6.0, 0.45, size), 2)
    points_balance = rng.integers(0, 80_001, size)
    service_issues = rng.binomial(3, 0.18, size)
    engagement_score = rng.integers(20, 101, size)
    tiers = rng.choice(
        ["Company", "Topázio", "Safira", "Diamante", "Diamante Unique"],
        size=size,
        p=[0.45, 0.25, 0.17, 0.10, 0.03],
    )
    preferred_routes = rng.choice(
        ["VCP-CNF", "VCP-REC", "VCP-SDU", "VCP-POA", "REC-FOR", "CNF-SSA"],
        size=size,
    )

    # Alvo criado por regra probabilística apenas para demonstrar avaliação.
    logit = (
        -1.7
        + 0.24 * months_since_last_flight
        + 0.58 * service_issues
        - 0.018 * engagement_score
        - 0.07 * flights
    )
    churn_probability = 1 / (1 + np.exp(-logit))
    churned_synthetic = rng.binomial(1, churn_probability)

    return pd.DataFrame({
        "customer_id": [f"SYN-{index:05d}" for index in range(1, size + 1)],
        "flights_last_12m": flights,
        "flights_previous_12m": flights_previous,
        "months_since_last_flight": months_since_last_flight,
        "avg_ticket_brl": avg_ticket_brl,
        "points_balance": points_balance,
        "service_issues_last_12m": service_issues,
        "engagement_score": engagement_score,
        "loyalty_tier_synthetic": tiers,
        "preferred_route_synthetic": preferred_routes,
        "churned_synthetic": churned_synthetic,
        "data_type": "synthetic",
    })


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    passengers = generate_passengers()
    passengers.to_csv(OUTPUT_FILE, index=False)
    print(f"Base sintética criada: {OUTPUT_FILE} ({len(passengers)} registros)")


if __name__ == "__main__":
    main()
