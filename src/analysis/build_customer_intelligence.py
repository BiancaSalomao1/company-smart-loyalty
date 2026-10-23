"""Cria scores demonstrativos de valor, risco e ações para dados sintéticos."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PASSENGERS_FILE = PROJECT_ROOT / "data" / "gold" / "passageiros_sinteticos.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "gold" / "perfis_clientes_sinteticos.csv"
METRICS_FILE = PROJECT_ROOT / "data" / "gold" / "metricas_modelo_sintetico.json"

FEATURES = [
    "flights_last_12m",
    "flights_previous_12m",
    "months_since_last_flight",
    "avg_ticket_brl",
    "points_balance",
    "service_issues_last_12m",
    "engagement_score",
]


def fit_logistic_regression(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Treina um baseline logístico pequeno com NumPy, sem dados reais."""
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales[scales == 0] = 1
    normalized = (features - means) / scales
    design = np.c_[np.ones(len(normalized)), normalized]
    weights = np.zeros(design.shape[1])

    for _ in range(1_500):
        logits = np.clip(design @ weights, -30, 30)
        probabilities = 1 / (1 + np.exp(-logits))
        weights -= 0.08 * (design.T @ (probabilities - labels) / len(labels))

    return weights, means, scales


def predict_probability(features: np.ndarray, weights: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    design = np.c_[np.ones(len(features)), (features - means) / scales]
    return 1 / (1 + np.exp(-np.clip(design @ weights, -30, 30)))


def segment(row: pd.Series) -> str:
    if row["clv_simulado_12m_brl"] >= 7_000 and row["churn_risk_score"] < 0.45:
        return "Alto valor engajado"
    if row["churn_risk_score"] >= 0.60:
        return "Risco de evasão"
    return "Em desenvolvimento"


def recommend(row: pd.Series) -> tuple[str, str]:
    if row["segment"] == "Risco de evasão" and row["clv_simulado_12m_brl"] >= 5_000:
        return (
            "Contato prioritário e revisão da experiência",
            "Alto valor simulado combinado com risco estimado elevado; validar elegibilidade antes de ofertar benefícios.",
        )
    if row["segment"] == "Alto valor engajado":
        return (
            "Ação de reconhecimento e engajamento",
            "Perfil sintético com valor projetado alto e risco menor; comunicar benefícios conforme regras vigentes.",
        )
    return (
        "Comunicação de engajamento",
        "Perfil sintético sem prioridade crítica; testar conteúdo e medir resposta antes de escalar.",
    )


def explain_risk(row: pd.Series) -> str:
    factors = []
    if row["months_since_last_flight"] >= 6:
        factors.append("recência de voo elevada")
    if row["flights_last_12m"] < row["flights_previous_12m"]:
        factors.append("queda na frequência de voos")
    if row["service_issues_last_12m"] >= 2:
        factors.append("ocorrências de atendimento no período")
    if row["engagement_score"] < 50:
        factors.append("engajamento reduzido")
    return ", ".join(factors) if factors else "nenhum fator crítico acima do limiar demonstrativo"


def draft_message(row: pd.Series) -> str:
    if row["churn_risk_score"] >= 0.60:
        return "Olá! Notamos uma mudança recente no seu padrão de viagem. Podemos ajudar a planejar sua próxima experiência?"
    return "Olá! Confira as opções do programa de fidelidade disponíveis para sua próxima viagem."


def build_intelligence(passengers: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Treina/avalia em partições artificiais e gera o perfil demonstrativo."""
    if not set(FEATURES + ["churned_synthetic", "customer_id"]).issubset(passengers.columns):
        raise ValueError("A base sintética não contém as colunas necessárias.")

    train_mask = np.arange(len(passengers)) % 5 != 0
    train = passengers.loc[train_mask]
    test = passengers.loc[~train_mask]
    weights, means, scales = fit_logistic_regression(
        train[FEATURES].to_numpy(float), train["churned_synthetic"].to_numpy(float)
    )
    probabilities = predict_probability(passengers[FEATURES].to_numpy(float), weights, means, scales)
    test_probability = probabilities[~train_mask]
    test_prediction = (test_probability >= 0.5).astype(int)
    test_labels = test["churned_synthetic"].to_numpy(int)
    true_positive = int(((test_prediction == 1) & (test_labels == 1)).sum())
    predicted_positive = int((test_prediction == 1).sum())
    actual_positive = int((test_labels == 1).sum())

    profile = passengers.copy()
    profile["churn_risk_score"] = probabilities.round(4)
    profile["clv_simulado_12m_brl"] = (
        profile["flights_last_12m"] * profile["avg_ticket_brl"] * 1.35
    ).round(2)
    profile["segment"] = profile.apply(segment, axis=1)
    recommendations = profile.apply(recommend, axis=1, result_type="expand")
    profile[["recommended_action", "recommendation_rationale"]] = recommendations
    profile["risk_explanation"] = profile.apply(explain_risk, axis=1)
    profile["suggested_channel"] = np.where(
        profile["engagement_score"] >= 60, "canal digital preferencial", "contato assistido"
    )
    profile["draft_message"] = profile.apply(draft_message, axis=1)
    profile["estimated_action_cost_brl"] = np.where(
        profile["churn_risk_score"] >= 0.60, 45.0, 12.0
    )
    profile["model_scope"] = "demonstração em dados sintéticos"

    metrics = {
        "data_type": "synthetic",
        "model": "logistic_regression_numpy_baseline",
        "test_records": int(len(test)),
        "accuracy": round(float((test_prediction == test_labels).mean()), 4),
        "precision": round(true_positive / predicted_positive, 4) if predicted_positive else 0.0,
        "recall": round(true_positive / actual_positive, 4) if actual_positive else 0.0,
        "limitation": "Métricas obtidas somente em alvo artificial; não validam risco de clientes reais.",
    }
    return profile, metrics


def main() -> None:
    if not PASSENGERS_FILE.exists():
        raise FileNotFoundError("Execute generate_passengers.py antes deste script.")
    profiles, metrics = build_intelligence(pd.read_csv(PASSENGERS_FILE))
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(OUTPUT_FILE, index=False)
    METRICS_FILE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Perfis criados: {OUTPUT_FILE} ({len(profiles)} registros)")
    print(f"Métricas: {METRICS_FILE}")


if __name__ == "__main__":
    main()
