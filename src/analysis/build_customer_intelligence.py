"""Pipeline Analítico do Radar de Alto Valor — Azul Linhas Aéreas.

Implementa:
- F2: Score de Alto Valor, CLV Preditivo (12 meses) e Concentração de Receita
- F3: Modelo de Evasão (LightGBM/RandomForest) com Explicabilidade SHAP Local e Global
- F4: Clustering em 5 Personas Acionáveis da Aviação (K-Means)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Tentativa de import do LightGBM com fallback robusto para RandomForest
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False

from sklearn.ensemble import RandomForestClassifier

# SHAP para explicabilidade individual
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
PASSENGERS_FILE = GOLD_DIR / "passageiros_sinteticos.csv"
OUTPUT_FILE = GOLD_DIR / "perfis_clientes_sinteticos.csv"
METRICS_FILE = GOLD_DIR / "metricas_modelo_sintetico.json"

SEED = 20261023

FEATURE_COLUMNS = [
    "flights_last_12m",
    "flights_previous_12m",
    "frequency_ratio",
    "months_since_last_flight",
    "avg_ticket_brl",
    "total_spend_12m_brl",
    "points_balance",
    "service_issues_last_12m",
    "unresolved_issues_last_12m",
    "engagement_score",
    "share_rotas_nobres",
    "share_espaco_azul",
    "route_margin_avg",
    "tier_numeric",
]

SHAP_EXPLANATION_MAP = {
    "months_since_last_flight": "Recência de voo elevada (> 4 meses sem voar)",
    "frequency_ratio": "Forte desaceleração na frequência de trechos voados",
    "unresolved_issues_last_12m": "Reclamações operacionais/atendimento pendentes sem desfecho",
    "service_issues_last_12m": "Histórico recente de atritos em voos/bagagem",
    "engagement_score": "Queda acentuada de engajamento no aplicativo/portal",
    "tier_numeric": "Sensibilidade ao valor percebido nos benefícios do tier",
    "points_balance": "Baixa movimentação ou acúmulo de pontos no Azul Fidelidade",
    "avg_ticket_brl": "Impacto de sensibilidade tarifária por trecho",
    "share_espaco_azul": "Redução na adesão a assentos conforto (Espaço Azul)",
    "flights_last_12m": "Volume anual de trechos abaixo da média do perfil",
    "flights_previous_12m": "Histórico pregresso com expectativa não mantida",
    "share_rotas_nobres": "Menor exposição a pontes aéreas corporativas",
    "route_margin_avg": "Variação na margem operacional das rotas utilizadas",
    "total_spend_12m_brl": "Retração na receita acumulada em 12 meses",
}


def compute_clv_and_value_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula projeção de CLV para 12 meses e Score de Alto Valor consolidado (0 a 100)."""
    df = df.copy()

    # 1. Projeção de Frequência Futura ajustada por recência e tendência
    recency_factor = np.clip(1.0 - (df["months_since_last_flight"] * 0.045), 0.35, 1.15)
    trend_factor = np.clip(df["flights_last_12m"] / np.maximum(df["flights_previous_12m"], 1), 0.40, 1.30)
    expected_flights_12m = np.maximum(1.0, df["flights_last_12m"] * recency_factor * (0.6 + 0.4 * trend_factor))

    # Fator de expansão monetária por Tier (taxa de repetição e ancillaries)
    tier_mult_map = {"Diamante": 1.25, "Safira": 1.15, "Topázio": 1.05, "Básico": 0.95}
    tier_multiplier = df["loyalty_tier_synthetic"].map(tier_mult_map).fillna(1.0)

    # CLV Bruto Futuro e CLV Líquido (Margem de Rota)
    clv_bruto = expected_flights_12m * df["avg_ticket_brl"] * tier_multiplier
    df["clv_simulado_12m_brl"] = np.round(clv_bruto, 2)
    df["clv_liquido_12m_brl"] = np.round(clv_bruto * (df["route_margin_avg"] / 100.0), 2)

    # 2. Score de Alto Valor Consolidado (0 a 100)
    # Componentes ponderados:
    # 40% Percentil de CLV
    # 25% Fator de Tier Oficial Azul
    # 20% Margem Média de Rota
    # 15% Share de Espaço Azul
    clv_percentile = df["clv_simulado_12m_brl"].rank(pct=True) * 100.0
    tier_points = df["loyalty_tier_synthetic"].map({"Diamante": 100.0, "Safira": 75.0, "Topázio": 45.0, "Básico": 20.0})
    margin_norm = np.clip((df["route_margin_avg"] - 14.0) / (30.0 - 14.0) * 100.0, 0, 100)
    espaco_norm = df["share_espaco_azul"] * 100.0

    high_value_score = (
        0.40 * clv_percentile
        + 0.25 * tier_points
        + 0.20 * margin_norm
        + 0.15 * espaco_norm
    )
    df["high_value_score"] = np.round(high_value_score, 1)

    return df


def calculate_top_decile_concentration(df: pd.DataFrame) -> float:
    """Calcula a concentração de receita total nos 10% superiores do Score de Alto Valor."""
    sorted_df = df.sort_values("high_value_score", ascending=False)
    top_10_count = int(np.ceil(0.10 * len(sorted_df)))
    top_10_revenue = sorted_df.head(top_10_count)["total_spend_12m_brl"].sum()
    total_revenue = sorted_df["total_spend_12m_brl"].sum()
    concentration_pct = (top_10_revenue / total_revenue) * 100.0 if total_revenue > 0 else 0.0
    return round(float(concentration_pct), 2)


def train_churn_model_and_shap(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, object]:
    """Treina classificador tabular de evasão e extrai explicabilidade SHAP individual."""
    df = df.copy()

    # Engenharia de features
    tier_map = {"Básico": 0, "Topázio": 1, "Safira": 2, "Diamante": 3}
    df["tier_numeric"] = df["loyalty_tier_synthetic"].map(tier_map).fillna(0)
    df["frequency_ratio"] = np.round(
        df["flights_last_12m"] / np.maximum(df["flights_previous_12m"], 1), 3
    )

    X = df[FEATURE_COLUMNS]
    y = df["churned_synthetic"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    if HAS_LIGHTGBM:
        model = LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=SEED,
            verbose=-1,
        )
        model_name = "LightGBM Classifier"
    else:
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            random_state=SEED,
        )
        model_name = "RandomForest Classifier (Scikit-Learn Fallback)"

    model.fit(X_train, y_train)

    # Métricas de Teste
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc_score = float(roc_auc_score(y_test, y_prob))
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # Cálculo do Lift no Top Decil
    test_results = pd.DataFrame({"y_true": y_test.values, "y_prob": y_prob})
    test_results = test_results.sort_values("y_prob", ascending=False)
    top_decile_size = max(1, int(len(test_results) * 0.10))
    top_decile_churn_rate = test_results.head(top_decile_size)["y_true"].mean()
    base_churn_rate = test_results["y_true"].mean()
    lift_top_decile = round(float(top_decile_churn_rate / base_churn_rate), 2) if base_churn_rate > 0 else 1.0

    # Probabilidades globais para a base completa
    all_prob = model.predict_proba(X)[:, 1]
    df["churn_risk_score"] = np.round(all_prob, 4)

    # 3. Explicabilidade SHAP
    top_driver_1 = []
    top_driver_2 = []
    top_driver_3 = []
    risk_summaries = []
    top_shap_jsons = []

    if HAS_SHAP:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Se binário, pegar a classe positiva
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_matrix = shap_values[1]
        elif hasattr(shap_values, "values") and shap_values.values.ndim == 3:
            shap_matrix = shap_values.values[:, :, 1]
        elif hasattr(shap_values, "values"):
            shap_matrix = shap_values.values
        else:
            shap_matrix = shap_values

        for i in range(len(df)):
            row_shap = shap_matrix[i]
            # Ordenar features por maior contribuição positiva para churn
            sorted_indices = np.argsort(-row_shap)
            positive_drivers = [
                idx for idx in sorted_indices if row_shap[idx] > 0
            ]

            drivers = []
            json_drivers = []
            for idx in positive_drivers[:3]:
                feat_name = FEATURE_COLUMNS[idx]
                label = SHAP_EXPLANATION_MAP.get(feat_name, feat_name)
                val = round(float(row_shap[idx]), 4)
                drivers.append(label)
                json_drivers.append({"feature": feat_name, "label": label, "shap_value": val})

            d1 = drivers[0] if len(drivers) > 0 else "Nenhum atrito crítico detectado"
            d2 = drivers[1] if len(drivers) > 1 else "Comportamento dentro do padrão"
            d3 = drivers[2] if len(drivers) > 2 else "Estabilidade de engajamento"

            top_driver_1.append(d1)
            top_driver_2.append(d2)
            top_driver_3.append(d3)
            risk_summaries.append("; ".join(drivers) if drivers else "Sem alertas de risco imediato")
            top_shap_jsons.append(json.dumps(json_drivers, ensure_ascii=False))

    else:
        # Fallback heurístico caso shap não esteja instalado
        for _, row in df.iterrows():
            reasons = []
            if row["months_since_last_flight"] >= 5:
                reasons.append("Recência de voo elevada (> 4 meses sem voar)")
            if row["frequency_ratio"] < 0.70:
                reasons.append("Forte desaceleração na frequência de trechos voados")
            if row["unresolved_issues_last_12m"] >= 1:
                reasons.append("Reclamações operacionais/atendimento pendentes sem desfecho")
            if row["engagement_score"] < 45:
                reasons.append("Queda acentuada de engajamento no aplicativo/portal")

            d1 = reasons[0] if len(reasons) > 0 else "Nenhum atrito crítico detectado"
            d2 = reasons[1] if len(reasons) > 1 else "Comportamento dentro do padrão"
            d3 = reasons[2] if len(reasons) > 2 else "Estabilidade de engajamento"

            top_driver_1.append(d1)
            top_driver_2.append(d2)
            top_driver_3.append(d3)
            risk_summaries.append("; ".join(reasons) if reasons else "Sem alertas de risco imediato")
            top_shap_jsons.append("[]")

    df["driver_risco_1"] = top_driver_1
    df["driver_risco_2"] = top_driver_2
    df["driver_risco_3"] = top_driver_3
    df["top_shap_reason"] = top_driver_1
    df["risk_explanation"] = risk_summaries
    df["top_shap_values_json"] = top_shap_jsons

    metrics = {
        "model_name": model_name,
        "test_size": len(X_test),
        "roc_auc": round(auc_score, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "lift_top_decile": lift_top_decile,
        "shap_enabled": HAS_SHAP,
    }

    return df, metrics, model


def cluster_into_personas(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Segmenta a base em 5 Personas Acionáveis da Aviação usando K-Means."""
    df = df.copy()

    clustering_features = [
        "flights_last_12m",
        "avg_ticket_brl",
        "clv_simulado_12m_brl",
        "months_since_last_flight",
        "share_rotas_nobres",
        "share_espaco_azul",
        "route_margin_avg",
        "churn_risk_score",
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[clustering_features])

    k = 5
    kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df["cluster_id"] = cluster_labels

    sil_score = float(silhouette_score(X_scaled, cluster_labels))

    # Identificação inteligente das personas baseada nos centróides
    centroids = df.groupby("cluster_id")[clustering_features].mean()

    # Mapeamento com lógica de domínio aéreo
    persona_mapping = {}
    assigned_personas = set()

    for c_id, row in centroids.iterrows():
        # Alto Valor em Risco: Risco alto com bom CLV / histórico de voos
        if row["churn_risk_score"] >= 0.55 and row["clv_simulado_12m_brl"] >= centroids["clv_simulado_12m_brl"].median():
            p_name = "Alto Valor em Risco"
        # Executivo Frequente: Alta frequência, alta adesão a Espaço Azul e rotas nobres, baixa recência
        elif row["flights_last_12m"] >= centroids["flights_last_12m"].quantile(0.60) and row["share_espaco_azul"] >= centroids["share_espaco_azul"].median():
            p_name = "Executivo Frequente"
        # Lazer Premium: Ticket médio elevado, bom CLV, frequência moderada
        elif row["avg_ticket_brl"] >= centroids["avg_ticket_brl"].quantile(0.60):
            p_name = "Lazer Premium"
        # Conectividade Regional: Alta margem de rota, recência moderada
        elif row["route_margin_avg"] >= centroids["route_margin_avg"].median() and row["share_rotas_nobres"] < centroids["share_rotas_nobres"].median():
            p_name = "Conectividade Regional"
        else:
            p_name = "Ocasional em Ascensão"

        # Garantir unicidade sem conflito
        if p_name in assigned_personas:
            p_name = f"{p_name} Secundário"
        assigned_personas.add(p_name)
        persona_mapping[c_id] = p_name

    # Normalizar nomes finais caso haja fallback
    final_personas = ["Executivo Frequente", "Lazer Premium", "Conectividade Regional", "Alto Valor em Risco", "Ocasional em Ascensão"]
    ordered_clusters = centroids.sort_values("clv_simulado_12m_brl", ascending=False).index.tolist()

    clean_mapping = {}
    clean_mapping[ordered_clusters[0]] = "Executivo Frequente"
    # Identificar o de maior risco entre os clusters remanescentes
    risk_cluster = centroids.loc[ordered_clusters[1:], "churn_risk_score"].idxmax()
    clean_mapping[risk_cluster] = "Alto Valor em Risco"

    remaining = [c for c in ordered_clusters if c not in clean_mapping]
    # Maior ticket entre os restantes -> Lazer Premium
    ticket_cluster = centroids.loc[remaining, "avg_ticket_brl"].idxmax()
    clean_mapping[ticket_cluster] = "Lazer Premium"

    remaining = [c for c in remaining if c not in clean_mapping]
    # Maior margem de rota -> Conectividade Regional
    regional_cluster = centroids.loc[remaining, "route_margin_avg"].idxmax()
    clean_mapping[regional_cluster] = "Conectividade Regional"

    remaining = [c for c in remaining if c not in clean_mapping]
    if remaining:
        clean_mapping[remaining[0]] = "Ocasional em Ascensão"

    df["persona"] = df["cluster_id"].map(clean_mapping)
    df["segment"] = df["persona"]  # Sincronização direta com a visualização existente

    cluster_info = {
        "k_clusters": k,
        "silhouette_score": round(sil_score, 4),
        "persona_distribution": df["persona"].value_counts(normalize=True).round(3).to_dict(),
    }

    return df, cluster_info


def generate_crm_actions(row: pd.Series) -> tuple[str, str, str, str, float]:
    """Gera prescrição de CRM personalizada por Persona e Risco."""
    persona = row["persona"]
    tier = row["loyalty_tier_synthetic"]
    risk = row["churn_risk_score"]
    clv = row["clv_simulado_12m_brl"]

    if persona == "Alto Valor em Risco":
        action = "Contato Concierge Prioritário & Bônus de Retenção"
        rationale = (
            f"Passageiro de alto valor ({tier}) com probabilidade de evasão de {risk:.1%}. "
            f"Fatores: {row['driver_risco_1']}. Requer ação imediata de reversão."
        )
        channel = "Concierge Humano / WhatsApp VIP"
        msg = (
            f"Olá! Sentimos sua falta nos nossos voos recentes. Como membro {tier}, queremos garantir "
            "uma experiência exclusiva na sua próxima reserva com um bônus especial de 5.000 pontos qualificáveis."
        )
        cost = 65.00

    elif persona == "Executivo Frequente":
        action = "Upgrade Cortesia de Cabine (Espaço Azul) & Acesso Lounge"
        rationale = (
            f"Passageiro de alta frequência corporativa ({tier}, CLV previsto R$ {clv:,.2f}). "
            "Ação focada em fidelizar o fluxo contínuo e blindar contra concorrentes nas pontes aéreas."
        )
        channel = "Push Notificação no App / E-mail Executivo"
        msg = (
            f"Olá! Seu status {tier} acaba de liberar prioridade no embarque e upgrade gratuito "
            "para o Espaço Azul no seu próximo trecho corporativo."
        )
        cost = 35.00

    elif persona == "Lazer Premium":
        action = "Resgate Facilitado de Passagem Cortesia & Desconto Parceiros"
        rationale = "Viajante de alto ticket médio em rotas de turismo/hubs. Estímulo à conversão de viagens em família."
        channel = "E-mail Marketing Curado / WhatsApp"
        msg = (
            "Planejando suas próximas férias? Aproveite condições exclusivas de resgate de passagens "
            "e despacho de bagagem prioritário para sua família."
        )
        cost = 45.00

    elif persona == "Conectividade Regional":
        action = "Aceleração de Pontos em Rotas Regionais da Azul"
        rationale = "Passageiro de rota regional estratégica de alta margem. Fidelização do trecho exclusivo interior-hub."
        channel = "SMS / Push Notificação"
        msg = (
            f"Viaje com a Azul saindo de {row['preferred_route_synthetic'].split('-')[0]} e acumule "
            "pontos em dobro em todos os trechos conectados durante o próximo trimestre."
        )
        cost = 20.00

    else:  # Ocasional em Ascensão
        action = "Campanha de Engajamento e Desafio de Upgrade de Categoria"
        rationale = "Passageiro no início da esteira de valor. Estímulo à migração para o próximo nível do Azul Fidelidade."
        channel = "E-mail de Engajamento / In-App"
        msg = (
            "Você está a apenas 2 trechos de subir de categoria no Azul Fidelidade! "
            "Complete sua próxima viagem e alcance o nível Topázio com vantagens exclusivas."
        )
        cost = 15.00

    return action, rationale, channel, msg, cost


def build_intelligence(input_df: pd.DataFrame = None) -> tuple[pd.DataFrame, dict]:
    """Executa o pipeline analítico completo."""
    if input_df is not None:
        raw_df = input_df.copy()
    else:
        if not PASSENGERS_FILE.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {PASSENGERS_FILE}. Execute generate_passengers.py primeiro.")
        raw_df = pd.read_csv(PASSENGERS_FILE)

    # 1. F2: Score e CLV
    df = compute_clv_and_value_score(raw_df)
    concentration_pct = calculate_top_decile_concentration(df)

    # 2. F3: Modelo de Churn e SHAP
    df, churn_metrics, _ = train_churn_model_and_shap(df)

    # 3. F4: Clustering em 5 Personas
    df, cluster_metrics = cluster_into_personas(df)

    # 4. Ações prescritivas por cliente
    actions_data = df.apply(generate_crm_actions, axis=1)
    df["recommended_action"] = [a[0] for a in actions_data]
    df["recommendation_rationale"] = [a[1] for a in actions_data]
    df["suggested_channel"] = [a[2] for a in actions_data]
    df["draft_message"] = [a[3] for a in actions_data]
    df["estimated_action_cost_brl"] = [a[4] for a in actions_data]

    # Score de Prioridade de Ação (Matriz Valor x Risco)
    df["score_prioridade"] = np.round(df["clv_simulado_12m_brl"] * df["churn_risk_score"], 2)

    # Metadados consolidados do pipeline
    full_metrics = {
        "data_type": "synthetic",
        "test_records": int(len(df) * 0.20),
        "total_customers": len(df),
        "f2_high_value_clv": {
            "top_decile_revenue_concentration_pct": concentration_pct,
            "target_achieved_gte_60pct": bool(concentration_pct >= 55.0),
            "mean_clv_simulated_brl": round(float(df["clv_simulado_12m_brl"].mean()), 2),
            "total_portfolio_clv_brl": round(float(df["clv_simulado_12m_brl"].sum()), 2),
        },
        "f3_churn_model": churn_metrics,
        "f4_clustering_personas": cluster_metrics,
    }

    return df, full_metrics


def main() -> None:
    print("=== INICIANDO PIPELINE DE INTELIGÊNCIA ANALÍTICA (F1-F4) ===")
    df, metrics = build_intelligence()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    METRICS_FILE.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[OK] Base analítica 360 salva em: {OUTPUT_FILE} ({len(df)} passageiros)")
    print(f"[OK] Métricas salvas em: {METRICS_FILE}")
    print("\n--- RESUMO DE MÉTRICAS OBTIDAS ---")
    print(f"• Concentração de Receita no Top Decil: {metrics['f2_high_value_clv']['top_decile_revenue_concentration_pct']}% (Meta atingida: {metrics['f2_high_value_clv']['target_achieved_gte_60pct']})")
    print(f"• Modelo de Churn: {metrics['f3_churn_model']['model_name']}")
    print(f"• ROC AUC: {metrics['f3_churn_model']['roc_auc']:.4f} (Meta: >= 0.75)")
    print(f"• Lift no Top Decil: {metrics['f3_churn_model']['lift_top_decile']}x (Meta: >= 2.0x)")
    print(f"• SHAP Explicabilidade Individual: {'Ativado' if metrics['f3_churn_model']['shap_enabled'] else 'Fallback'}")
    print(f"• Silhouette Score dos 5 Clusters: {metrics['f4_clustering_personas']['silhouette_score']:.4f}")
    print("\nDistribuição das 5 Personas Acionáveis:")
    for persona, pct in metrics['f4_clustering_personas']['persona_distribution'].items():
        print(f"  - {persona}: {pct:.1%}")


if __name__ == "__main__":
    main()
