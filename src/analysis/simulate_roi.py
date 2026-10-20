
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "config" / "roi_scenarios.yaml"
OUTPUT_FILE = PROJECT_ROOT / "data" / "gold" / "roi_scenarios.csv"

with CONFIG_FILE.open(encoding="utf-8") as file:
    config = yaml.safe_load(file)

results = []

for scenario in config["scenarios"]:
    units = float(scenario["eligible_units"])
    unit_cost = float(scenario["action_cost_per_unit"])
    fixed_cost = float(scenario["fixed_campaign_cost"])
    retention_pct = float(
        scenario["assumed_incremental_retention_pct"]
    )
    margin = float(
        scenario["contribution_margin_per_retained_customer"]
    )
    horizon = int(scenario["benefit_horizon_months"])

    if units < 0 or unit_cost < 0 or fixed_cost < 0:
        raise ValueError(f"Custos ou unidades negativos: {scenario['id']}")

    if not 0 <= retention_pct <= 100:
        raise ValueError(f"Percentual de retenção inválido: {scenario['id']}")

    if margin < 0 or horizon <= 0:
        raise ValueError(f"Margem ou horizonte inválido: {scenario['id']}")

    investment = units * unit_cost + fixed_cost
    expected_retained = units * retention_pct / 100
    benefit = expected_retained * margin
    net_benefit = benefit - investment

    roi_pct = (
        net_benefit / investment * 100
        if investment > 0
        else None
    )

    monthly_benefit = benefit / horizon
    payback_months = (
        investment / monthly_benefit
        if monthly_benefit > 0
        else None
    )

    results.append({
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "data_type": "simulado",
        "eligible_units_assumption": units,
        "action_cost_per_unit_brl": unit_cost,
        "fixed_campaign_cost_brl": fixed_cost,
        "assumed_incremental_retention_pct": retention_pct,
        "contribution_margin_per_retained_customer_brl": margin,
        "benefit_horizon_months": horizon,
        "investment_brl": investment,
        "expected_retained_units_simulated": expected_retained,
        "gross_incremental_benefit_brl": benefit,
        "net_benefit_brl": net_benefit,
        "roi_pct": roi_pct,
        "payback_months": payback_months,
        "retention_validated": False,
    })

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("\n=== SIMULAÇÃO DE ROI ===")
print(
    result_df[
        [
            "scenario_name",
            "investment_brl",
            "gross_incremental_benefit_brl",
            "net_benefit_brl",
            "roi_pct",
            "payback_months",
        ]
    ].to_string(index=False)
)
print(f"\nArquivo gerado: {OUTPUT_FILE}")
print("ATENÇÃO: todos os resultados são simulados, não observados.")
