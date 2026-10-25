"""Gera uma base fictícia e calibrada da Azul Linhas Aéreas para demonstrar o MVP de CRM."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
PASSENGERS_FILE = GOLD_DIR / "passageiros_sinteticos.csv"
ROUTES_FILE = GOLD_DIR / "dim_rotas_azul.csv"
TRANSACTIONS_FILE = GOLD_DIR / "fact_transacoes_sinteticas.csv"

SEED = 20261023

# Catálogo oficial de rotas da malha Azul com margens estimadas
AZUL_ROUTES = [
    # Pontes Aéreas e Rotas de Alta Densidade / Corporativas (Margem média ~24-25%)
    {"origem": "CGH", "destino": "SDU", "tipo_rota": "Ponte Aérea", "margem_estimada_pct": 25.0, "peso": 0.12},
    {"origem": "VCP", "destino": "SDU", "tipo_rota": "Ponte Aérea", "margem_estimada_pct": 24.0, "peso": 0.10},
    {"origem": "VCP", "destino": "BSB", "tipo_rota": "Ponte Aérea", "margem_estimada_pct": 23.0, "peso": 0.08},
    # Conexões Hub-a-Hub (Margem média ~19-21%)
    {"origem": "VCP", "destino": "CNF", "tipo_rota": "Hub Principal", "margem_estimada_pct": 19.5, "peso": 0.14},
    {"origem": "VCP", "destino": "REC", "tipo_rota": "Hub Principal", "margem_estimada_pct": 21.0, "peso": 0.12},
    {"origem": "CNF", "destino": "REC", "tipo_rota": "Hub Principal", "margem_estimada_pct": 20.0, "peso": 0.08},
    {"origem": "VCP", "destino": "POA", "tipo_rota": "Hub Principal", "margem_estimada_pct": 19.0, "peso": 0.07},
    {"origem": "VCP", "destino": "CWB", "tipo_rota": "Hub Principal", "margem_estimada_pct": 18.5, "peso": 0.07},
    {"origem": "CNF", "destino": "SSA", "tipo_rota": "Hub Principal", "margem_estimada_pct": 19.5, "peso": 0.06},
    # Rotas Regionais Conectadas (Margem superior ~26-28% por menor sobreposição concorrencial)
    {"origem": "VCP", "destino": "RAO", "tipo_rota": "Regional Conectado", "margem_estimada_pct": 28.0, "peso": 0.04},
    {"origem": "CNF", "destino": "MOC", "tipo_rota": "Regional Conectado", "margem_estimada_pct": 27.5, "peso": 0.03},
    {"origem": "REC", "destino": "JDO", "tipo_rota": "Regional Conectado", "margem_estimada_pct": 26.5, "peso": 0.03},
    {"origem": "REC", "destino": "PNZ", "tipo_rota": "Regional Conectado", "margem_estimada_pct": 26.0, "peso": 0.03},
    {"origem": "VCP", "destino": "CAC", "tipo_rota": "Regional Conectado", "margem_estimada_pct": 28.5, "peso": 0.02},
    {"origem": "VCP", "destino": "UDI", "tipo_rota": "Regional Conectado", "margem_estimada_pct": 27.0, "peso": 0.02},
]


def generate_routes_dim() -> pd.DataFrame:
    """Gera a dimensão analítica de rotas da Azul."""
    routes_df = pd.DataFrame(AZUL_ROUTES)[["origem", "destino", "tipo_rota", "margem_estimada_pct"]]
    return routes_df


def generate_passengers_and_transactions(
    size: int = 1_500,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Gera base de clientes calibrada e transações respeitando os 4 tiers oficiais da Azul."""
    if size < 1:
        raise ValueError("size deve ser maior que zero.")

    rng = np.random.default_rng(seed)

    # 1. Dimensão de rotas
    routes_df = generate_routes_dim()
    route_keys = [f"{r['origem']}-{r['destino']}" for r in AZUL_ROUTES]
    route_weights = np.array([r["peso"] for r in AZUL_ROUTES])
    route_weights = route_weights / route_weights.sum()
    route_margin_map = {f"{r['origem']}-{r['destino']}": r["margem_estimada_pct"] for r in AZUL_ROUTES}
    route_type_map = {f"{r['origem']}-{r['destino']}": r["tipo_rota"] for r in AZUL_ROUTES}

    # 2. Distribuição calibrada dos 4 Tiers oficiais do Azul Fidelidade
    tier_names = ["Básico", "Topázio", "Safira", "Diamante"]
    tier_probs = [0.55, 0.22, 0.13, 0.10]
    tiers = rng.choice(tier_names, size=size, p=tier_probs)

    # Vetores de suporte indexados
    flights_12m = np.zeros(size, dtype=int)
    flights_prev_12m = np.zeros(size, dtype=int)
    months_recency = np.zeros(size, dtype=int)
    avg_ticket = np.zeros(size, dtype=float)
    points_balance = np.zeros(size, dtype=int)
    share_espaco_azul = np.zeros(size, dtype=float)
    service_issues = np.zeros(size, dtype=int)
    unresolved_issues = np.zeros(size, dtype=int)
    engagement_score = np.zeros(size, dtype=int)
    preferred_routes = []
    preferred_route_types = []
    route_margin_avg = np.zeros(size, dtype=float)
    share_rotas_nobres = np.zeros(size, dtype=float)

    for i in range(size):
        tier = tiers[i]
        pref_route = rng.choice(route_keys, p=route_weights)
        preferred_routes.append(pref_route)
        pref_type = route_type_map[pref_route]
        preferred_route_types.append(pref_type)

        # Calibração comportamental estrita por Tier (Concentração Pareto Real da Aviação)
        if tier == "Diamante":
            # Viajante frequente corporativo / alta renda (Top Decil)
            f_12m = int(rng.poisson(28) + rng.integers(6, 18))
            f_prev = int(np.maximum(16, f_12m + rng.integers(-10, 8)))
            recency = int(rng.choice([0, 1, 2, 3, 5, 8], p=[0.45, 0.25, 0.15, 0.07, 0.05, 0.03]))
            ticket = np.round(rng.normal(2150, 320), 2)
            pts = int(rng.integers(80_000, 320_000))
            espaco_az = np.round(rng.uniform(0.65, 0.98), 2)
            eng = int(rng.integers(70, 100))
            nobres = np.round(rng.uniform(0.70, 0.98), 2)
        elif tier == "Safira":
            # Viajante regular qualificado
            f_12m = int(rng.poisson(8) + rng.integers(1, 5))
            f_prev = int(np.maximum(4, f_12m + rng.integers(-5, 4)))
            recency = int(rng.choice([0, 1, 2, 4, 6, 9], p=[0.30, 0.30, 0.20, 0.10, 0.06, 0.04]))
            ticket = np.round(rng.normal(760, 110), 2)
            pts = int(rng.integers(22_000, 75_000))
            espaco_az = np.round(rng.uniform(0.25, 0.60), 2)
            eng = int(rng.integers(50, 85))
            nobres = np.round(rng.uniform(0.40, 0.75), 2)
        elif tier == "Topázio":
            # Viajante em ascensão
            f_12m = int(rng.poisson(4) + 1)
            f_prev = int(np.maximum(2, f_12m + rng.integers(-3, 3)))
            recency = int(rng.choice([1, 2, 3, 5, 7, 10], p=[0.20, 0.25, 0.25, 0.15, 0.10, 0.05]))
            ticket = np.round(rng.normal(520, 75), 2)
            pts = int(rng.integers(6_000, 25_000))
            espaco_az = np.round(rng.uniform(0.08, 0.30), 2)
            eng = int(rng.integers(35, 75))
            nobres = np.round(rng.uniform(0.25, 0.55), 2)
        else:  # Básico
            # Viajante ocasional / lazer esporádico
            f_12m = int(rng.choice([1, 2, 3], p=[0.60, 0.30, 0.10]))
            f_prev = int(np.maximum(1, f_12m + rng.integers(-2, 2)))
            recency = int(rng.integers(1, 13))
            ticket = np.round(rng.normal(410, 55), 2)
            pts = int(rng.integers(500, 9_000))
            espaco_az = np.round(rng.uniform(0.0, 0.12), 2)
            eng = int(rng.integers(15, 60))
            nobres = np.round(rng.uniform(0.10, 0.40), 2)

        # Ajuste de margem média da rota do cliente com leve dispersão realista
        base_margin = route_margin_map[pref_route]
        margin_sample = np.clip(rng.normal(base_margin, 1.2), 14.0, 32.0)

        # Ocorrências de atendimento e atritos
        issues = int(rng.choice([0, 1, 2, 3, 4], p=[0.60, 0.24, 0.10, 0.04, 0.02]))
        unresolved = int(rng.binomial(issues, 0.45)) if issues > 0 else 0

        flights_12m[i] = max(1, f_12m)
        flights_prev_12m[i] = max(0, f_prev)
        months_recency[i] = recency
        avg_ticket[i] = max(250.0, ticket)
        points_balance[i] = pts
        share_espaco_azul[i] = espaco_az
        service_issues[i] = issues
        unresolved_issues[i] = unresolved
        engagement_score[i] = eng
        route_margin_avg[i] = np.round(margin_sample, 2)
        share_rotas_nobres[i] = nobres

    # 3. Modelagem do Alvo de Evasão Calibrado (Ground Truth para Treino Tabular)
    # Lógica estruturada: churn real decorre de queda de frequência, recência elevada,
    # atrito de serviço não resolvido, baixo engajamento digital e perda de incentivos.
    freq_drop = np.maximum(0, flights_prev_12m - flights_12m) / np.maximum(1, flights_prev_12m)
    recency_risk = np.where(months_recency >= 6, 1.2, np.where(months_recency >= 4, 0.5, -0.4))
    issue_risk = unresolved_issues * 0.95 + service_issues * 0.40
    engagement_factor = (50 - engagement_score) * 0.035
    tier_protection = np.select(
        [tiers == "Diamante", tiers == "Safira", tiers == "Topázio", tiers == "Básico"],
        [-0.45, -0.20, 0.05, 0.35],
    )

    latent_churn_score = (
        -1.35
        + 1.85 * freq_drop
        + 0.90 * recency_risk
        + 0.65 * issue_risk
        + engagement_factor
        + tier_protection
        + rng.normal(0, 0.35, size)  # Ruído empírico estocástico
    )

    churn_prob = 1 / (1 + np.exp(-latent_churn_score))
    churned_synthetic = (churn_prob >= 0.52).astype(int)

    # 4. Construção do DataFrame de Clientes
    customer_ids = [f"SYN-{idx:05d}" for idx in range(1, size + 1)]
    total_spend_12m = np.round(flights_12m * avg_ticket, 2)

    passengers_df = pd.DataFrame({
        "customer_id": customer_ids,
        "loyalty_tier_synthetic": tiers,
        "flights_last_12m": flights_12m,
        "flights_previous_12m": flights_prev_12m,
        "months_since_last_flight": months_recency,
        "avg_ticket_brl": avg_ticket,
        "total_spend_12m_brl": total_spend_12m,
        "points_balance": points_balance,
        "service_issues_last_12m": service_issues,
        "unresolved_issues_last_12m": unresolved_issues,
        "engagement_score": engagement_score,
        "preferred_route_synthetic": preferred_routes,
        "route_type_preferred": preferred_route_types,
        "route_margin_avg": route_margin_avg,
        "share_rotas_nobres": share_rotas_nobres,
        "share_espaco_azul": share_espaco_azul,
        "churned_synthetic": churned_synthetic,
        "data_type": "synthetic",
    })

    # 5. Geração de Transações Sintéticas Detalhadas (fact_transacoes_sinteticas)
    transactions = []
    base_date = date(2026, 10, 20)
    tx_counter = 1

    for idx, row in passengers_df.iterrows():
        c_id = row["customer_id"]
        n_flights = int(row["flights_last_12m"])
        rec_months = int(row["months_since_last_flight"])
        c_tier = row["loyalty_tier_synthetic"]
        c_ticket = float(row["avg_ticket_brl"])
        pref_r = row["preferred_route_synthetic"]
        origem, destino = pref_r.split("-")

        # Gerar datas ordenadas para os voos nos últimos 365 dias
        min_offset = min(max(rec_months * 30, 2), 340)
        flight_offsets = sorted(rng.integers(min_offset, 365, size=n_flights), reverse=True)
        if flight_offsets and rec_months <= 1:
            flight_offsets[-1] = int(rec_months * 30 + rng.integers(1, 15))

        for offset in flight_offsets:
            flight_date = base_date - timedelta(days=int(offset))
            # Variação no valor da tarifa
            fare = np.round(max(180.0, rng.normal(c_ticket, c_ticket * 0.15)), 2)
            # Pontos acumulados (regra aproximada: multiplicador por tier)
            mult = 4 if c_tier == "Diamante" else (3 if c_tier == "Safira" else 2)
            pts = int(fare * mult)

            # Classe de cabine
            if c_tier == "Diamante" and rng.random() < 0.65:
                cabine = "Espaco Azul"
            elif c_tier == "Safira" and rng.random() < 0.35:
                cabine = "Espaco Azul"
            else:
                cabine = "Economica"

            transactions.append({
                "transacao_id": f"TX-{tx_counter:07d}",
                "customer_id": c_id,
                "data_voo": flight_date.isoformat(),
                "origem": origem,
                "destino": destino,
                "tarifa_paga": fare,
                "pontos_acumulados": pts,
                "classe_cabine": cabine,
            })
            tx_counter += 1

    transactions_df = pd.DataFrame(transactions)

    return passengers_df, routes_df, transactions_df


def generate_passengers(size: int = 1500, seed: int = SEED) -> pd.DataFrame:
    """Gera DataFrame de passageiros com interface retrocompatível."""
    passengers_df, _, _ = generate_passengers_and_transactions(size=size, seed=seed)
    return passengers_df


def main() -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    passengers_df, routes_df, transactions_df = generate_passengers_and_transactions()

    passengers_df.to_csv(PASSENGERS_FILE, index=False, encoding="utf-8")
    routes_df.to_csv(ROUTES_FILE, index=False, encoding="utf-8")
    transactions_df.to_csv(TRANSACTIONS_FILE, index=False, encoding="utf-8")

    print(f"[OK] Base sintética gerada: {PASSENGERS_FILE} ({len(passengers_df)} clientes)")
    print(f"[OK] Dimensão de rotas gerada: {ROUTES_FILE} ({len(routes_df)} rotas)")
    print(f"[OK] Transações sintéticas geradas: {TRANSACTIONS_FILE} ({len(transactions_df)} trechos voados)")
    print("\nDistribuição por Tier:")
    print(passengers_df["loyalty_tier_synthetic"].value_counts(normalize=True).round(3))
    print(f"\nTaxa global de Churn Sintético: {passengers_df['churned_synthetic'].mean():.1%}")


if __name__ == "__main__":
    main()
