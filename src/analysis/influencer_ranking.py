"""Módulo de inteligência de relacionamento com criadores e influenciadores de viagens."""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFLUENCERS_GOLD_PATH = PROJECT_ROOT / "data" / "gold" / "influenciadores_viagem.csv"

# Dados públicos e critérios de curadoria transparente para a demonstração
DEFAULT_CREATORS = [
    {
        "channel_id": "UC_estevampelomundo",
        "creator_name": "Estevam Pelo Mundo",
        "focus_theme": "Milhas, Cartões e Aviação",
        "primary_platform": "YouTube / Instagram",
        "subscriber_tier": "Mega (1M+)",
        "content_alignment_score": 0.95,
        "loyalty_focus": "Alto (foco em resgate de pontos e emissões)",
        "avg_views_per_video": 85000,
        "estimated_cpm_brl": 120.0,
        "sample_video_topics": "Avaliação de Classe Executiva, Dicas de Status Match",
        "suggested_partnership": "Ação educativa sobre uso de pontos e benefícios VIP",
        "notes": "Público altamente qualificado em programas de fidelidade.",
    },
    {
        "channel_id": "UC_passageirodeprimeira",
        "creator_name": "Passageiro de Primeira",
        "focus_theme": "Fidelidade e Salas VIP",
        "primary_platform": "YouTube / Blog",
        "subscriber_tier": "Macro (500k-1M)",
        "content_alignment_score": 0.98,
        "loyalty_focus": "Muito Alto (referência técnica no Brasil)",
        "avg_views_per_video": 45000,
        "estimated_cpm_brl": 150.0,
        "sample_video_topics": "Como atingir nível Diamante, Análise de Lounges",
        "suggested_partnership": "Co-criação de conteúdo sobre acúmulo de pontos qualificáveis",
        "notes": "Alta densidade de passageiros frequentes de alta renda.",
    },
    {
        "channel_id": "UC_loucoporviagens",
        "creator_name": "Louco Por Viagens",
        "focus_theme": "Destinos Regionais e Experiências",
        "primary_platform": "YouTube / Instagram",
        "subscriber_tier": "Macro (500k-1M)",
        "content_alignment_score": 0.88,
        "loyalty_focus": "Médio (experiências e rotas de lazer)",
        "avg_views_per_video": 60000,
        "estimated_cpm_brl": 90.0,
        "sample_video_topics": "Roteiros no Nordeste, Melhores conexões",
        "suggested_partnership": "Campanha temática de turismo regional e conexões diretas",
        "notes": "Excelente alcance para conexões e rotas no interior/Nordeste.",
    },
    {
        "channel_id": "UC_aero_por_tras_da_aviacao",
        "creator_name": "Aero Por Trás da Aviação",
        "focus_theme": "Aeronaves e Experiência Operacional",
        "primary_platform": "YouTube",
        "subscriber_tier": "Mega (2M+)",
        "content_alignment_score": 0.82,
        "loyalty_focus": "Médio (bastidores, frotas e pontualidade)",
        "avg_views_per_video": 220000,
        "estimated_cpm_brl": 80.0,
        "sample_video_topics": "Como funciona o hub de Viracopos, Frota Embraer",
        "suggested_partnership": "Ação institucional de transparência operacional e pontualidade",
        "notes": "Forte credibilidade em assuntos operacionais e conforto a bordo.",
    },
    {
        "channel_id": "UC_melhoresdestinos",
        "creator_name": "Melhores Destinos",
        "focus_theme": "Promoções e Resgates",
        "primary_platform": "YouTube / App",
        "subscriber_tier": "Mega (1M+)",
        "content_alignment_score": 0.92,
        "loyalty_focus": "Alto (promoções de passagens e transferência de pontos)",
        "avg_views_per_video": 75000,
        "estimated_cpm_brl": 110.0,
        "sample_video_topics": "Alertas de promoções com pontos, Guias de destinos",
        "suggested_partnership": "Divulgação de campanhas de bônus de transferência de milhas",
        "notes": "Canal com conversão ágil para promoções de passagens com pontos.",
    },
]


def calculate_influencer_ranking(
    creators: list[dict[str, Any]] | None = None,
    weight_alignment: float = 0.50,
    weight_reach: float = 0.30,
    weight_cost_efficiency: float = 0.20,
) -> pd.DataFrame:
    """Calcula score transparente de adequação para parcerias.
    
    Critérios:
    - Alinhamento temático com o programa de fidelidade.
    - Alcance e engajamento médio.
    - Custo estimado por visualização (CPM).
    """
    data = creators or DEFAULT_CREATORS
    df = pd.DataFrame(data)

    max_views = df["avg_views_per_video"].max() or 1
    min_cpm = df["estimated_cpm_brl"].min() or 1
    max_cpm = df["estimated_cpm_brl"].max() or 1

    # Normalizações entre 0 e 1
    reach_norm = df["avg_views_per_video"] / max_views
    cost_efficiency = 1 - (df["estimated_cpm_brl"] - min_cpm) / (max_cpm - min_cpm + 1e-6)

    score = (
        df["content_alignment_score"] * weight_alignment
        + reach_norm * weight_reach
        + cost_efficiency * weight_cost_efficiency
    )

    df["fit_score"] = score.round(3)
    df["ranking"] = df["fit_score"].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("ranking", ascending=True).reset_index(drop=True)
    return df


def save_influencer_dataset(output_path: Path = INFLUENCERS_GOLD_PATH) -> pd.DataFrame:
    """Persiste a base ranqueada na camada Gold."""
    df = calculate_influencer_ranking()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return df


if __name__ == "__main__":
    ranked = save_influencer_dataset()
    print("=== RANKING DE INFLUENCIADORES E PARCEIROS ===")
    print(ranked[["ranking", "creator_name", "focus_theme", "fit_score", "suggested_partnership"]].to_string(index=False))
