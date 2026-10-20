
from pathlib import Path

import pandas as pd
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = PROJECT_ROOT / "data" / "gold"

INPUT_FILE = GOLD_DIR / "company_desfecho_por_problema_2025.csv"
OUTPUT_FILE = GOLD_DIR / "company_recomendacoes_2025.csv"


def recommend_action(problem: str) -> tuple[str, str]:
    """Retorna uma ação sugerida e a justificativa baseada no tema."""
    text = unicodedata.normalize(
    "NFKD", str(problem).casefold()
    ).encode("ascii", "ignore").decode("ascii")

    if any(word in text for word in ["cancelamento", "atraso", "embarque"]):
        return (
            "comunicacao_e_acompanhamento_operacional",
            "O problema envolve alteração ou execução da viagem; "
            "priorizar comunicação, orientação e acompanhamento do caso.",
        )

    if any(word in text for word in ["bagagem", "extravio", "avaria"]):
        return (
            "acompanhamento_de_bagagem",
            "O problema envolve bagagem; priorizar atualização de status "
            "e orientação sobre os procedimentos aplicáveis.",
        )

    if any(word in text for word in ["cobranca", "reembolso", "tarifa", "valor"]):
        return (
            "resolucao_financeira",
            "O problema envolve cobrança ou devolução de valores; "
            "priorizar análise do caso e comunicação clara do desfecho.",
        )

    if any(word in text for word in ["atendimento", "contato", "sac"]):
        return (
            "atendimento_especializado",
            "O problema envolve atendimento; priorizar encaminhamento "
            "adequado e prevenção de contatos repetidos.",
        )

    if any(word in text for word in ["fidelidade", "pontuacao", "pontos", "resgate"]):
        return (
            "suporte_ao_programa_de_fidelidade",
            "O problema envolve fidelidade ou pontuação; revisar regras "
            "aplicáveis e orientar o consumidor.",
        )

    return (
        "analise_e_tratamento_direcionado",
        "Revisar o caso e definir uma ação compatível com o problema identificado.",
    )


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: {INPUT_FILE}. "
        "Execute primeiro analyze_anac_2025.py."
    )

df = pd.read_csv(INPUT_FILE)

required = {
    "Grupo Problema",
    "Problema",
    "Não Resolvida",
    "Resolvida",
    "total_avaliado",
    "nao_resolvidas_pct",
}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Colunas ausentes no arquivo de entrada: {sorted(missing)}")

df["quantidade_nao_resolvida"] = df["Não Resolvida"].fillna(0).astype(int)
df["quantidade_resolvida"] = df["Resolvida"].fillna(0).astype(int)
df["total_avaliado"] = df["total_avaliado"].fillna(0).astype(int)

recommendations = df[
    df["total_avaliado"] >= 30
].copy()

actions = recommendations["Problema"].apply(recommend_action)
recommendations["acao_sugerida"] = actions.apply(lambda item: item[0])
recommendations["justificativa"] = actions.apply(lambda item: item[1])

recommendations["fonte"] = "ANAC - Consumidor.gov.br"
recommendations["ano_referencia"] = 2025
recommendations["tipo_evidencia"] = "dado_real_observado"
recommendations["retencao_incremental_validada"] = False
recommendations["custo_estimado_brl"] = pd.NA
recommendations["roi_estimado_pct"] = pd.NA

recommendations = recommendations.sort_values(
    ["nao_resolvidas_pct", "total_avaliado"],
    ascending=[False, False],
)

recommendations.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"Recomendações geradas: {len(recommendations)}")
print(f"Arquivo: {OUTPUT_FILE}")
print(
    recommendations[
        [
            "Problema",
            "total_avaliado",
            "nao_resolvidas_pct",
            "acao_sugerida",
        ]
    ].head(10).to_string(index=False)
)
