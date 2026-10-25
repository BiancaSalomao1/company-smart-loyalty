
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "bronze" / "anac"
    / "dadosconsumidor2025.csv"
)
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"

SILVER_DIR.mkdir(parents=True, exist_ok=True)
GOLD_DIR.mkdir(parents=True, exist_ok=True)

# Lê o CSV original: a primeira linha contém a data de atualização.
df = pd.read_csv(
    INPUT_FILE,
    sep=";",
    skiprows=1,
    quotechar='"',
    encoding="utf-8-sig",
    dtype=str,
    na_values=["null"],
    keep_default_na=False,
)

# Remove espaços acidentais dos nomes e valores textuais.
df.columns = df.columns.str.strip()

for column in df.select_dtypes(include=["object", "str"]).columns:
    df[column] = df[column].str.strip()

# Verificações básicas antes de analisar.
required_columns = [
    "Nome Fantasia",
    "Data Abertura",
    "Grupo Problema",
    "Problema",
    "Respondida",
    "Situação",
    "Avaliação Reclamação",
    "Nota do Consumidor",
    "Tempo Resposta (em dias)",
]

missing = [column for column in required_columns if column not in df.columns]
if missing:
    raise ValueError(f"Colunas esperadas não encontradas: {missing}")

# Filtra a companhia (Azul Linhas Aéreas) e salva os registros tratados na Silver.
company = df[df["Nome Fantasia"].str.contains("azul", case=False, na=False)].copy()
if company.empty:
    company = df[df["Nome Fantasia"].str.contains("company", case=False, na=False)].copy()

if company.empty:
    raise ValueError("Nenhum registro encontrado para Azul Linhas Aéreas.")

company["Data Abertura"] = pd.to_datetime(
    company["Data Abertura"], errors="coerce"
)
company["Nota do Consumidor"] = pd.to_numeric(
    company["Nota do Consumidor"], errors="coerce"
)
company["Tempo Resposta (em dias)"] = pd.to_numeric(
    company["Tempo Resposta (em dias)"], errors="coerce"
)

company.to_csv(
    SILVER_DIR / "company_reclamacoes_2025.csv",
    index=False,
    encoding="utf-8",
)


# Indicadores gerais: considerar apenas avaliações efetivas.
total = len(company)

avaliacoes_validas = company[
    company["Avaliação Reclamação"].isin(["Resolvida", "Não Resolvida"])
]

notas = pd.to_numeric(
    company["Nota do Consumidor"], errors="coerce"
).dropna()

tempos = company["Tempo Resposta (em dias)"].dropna()

nao_resolvidas = (
    avaliacoes_validas["Avaliação Reclamação"] == "Não Resolvida"
).sum()

summary = pd.DataFrame([{
    "ano": 2025,
    "empresa": "Azul Linhas Aéreas",
    "total_reclamacoes": total,
    "reclamacoes_respondidas_pct": (
        company["Respondida"].eq("S").mean() * 100
    ),
    "reclamacoes_avaliadas": len(avaliacoes_validas),
    "reclamacoes_resolvidas": (
        avaliacoes_validas["Avaliação Reclamação"] == "Resolvida"
    ).sum(),
    "reclamacoes_nao_resolvidas": nao_resolvidas,
    "nao_resolvidas_pct_sobre_avaliadas": (
        nao_resolvidas / len(avaliacoes_validas) * 100
        if len(avaliacoes_validas) else None
    ),
    "nota_media_consumidor": notas.mean() if len(notas) else None,
    "tempo_medio_resposta_dias": tempos.mean() if len(tempos) else None,
    "tempo_mediano_resposta_dias": tempos.median() if len(tempos) else None,
}])

summary.to_csv(
    GOLD_DIR / "company_indicadores_2025.csv", index=False
)

# Grupos de problemas: frequência e percentual sobre todas as reclamações.
problems = (
    company.groupby("Grupo Problema", dropna=False)
    .size()
    .reset_index(name="quantidade")
    .sort_values("quantidade", ascending=False)
)

problems["percentual"] = problems["quantidade"] / total * 100
problems.to_csv(
    GOLD_DIR / "company_grupos_problema_2025.csv", index=False
)

# Problemas detalhados mais frequentes.
details = (
    company.groupby(["Grupo Problema", "Problema"], dropna=False)
    .size()
    .reset_index(name="quantidade")
    .sort_values("quantidade", ascending=False)
)

details.to_csv(
    GOLD_DIR / "company_problemas_detalhados_2025.csv", index=False
)

# Cruzamento: grupo de problema x avaliação efetiva.
outcomes = (
    avaliacoes_validas
    .groupby(["Grupo Problema", "Avaliação Reclamação"], dropna=False)
    .size()
    .unstack(fill_value=0)
)

for column in ["Resolvida", "Não Resolvida"]:
    if column not in outcomes.columns:
        outcomes[column] = 0

outcomes["total_avaliado"] = (
    outcomes["Resolvida"] + outcomes["Não Resolvida"]
)

outcomes["nao_resolvidas_pct"] = (
    outcomes["Não Resolvida"]
    .div(outcomes["total_avaliado"].replace(0, float("nan")))
    * 100
)

outcomes = outcomes.reset_index().sort_values(
    "nao_resolvidas_pct", ascending=False
)

outcomes.to_csv(
    GOLD_DIR / "company_desfecho_por_grupo_2025.csv",
    index=False,
)

# Cruzamento por problema específico, útil para recomendar ações.
detail_outcomes = (
    avaliacoes_validas
    .groupby(
        ["Grupo Problema", "Problema", "Avaliação Reclamação"],
        dropna=False,
    )
    .size()
    .unstack(fill_value=0)
)

for column in ["Resolvida", "Não Resolvida"]:
    if column not in detail_outcomes.columns:
        detail_outcomes[column] = 0

detail_outcomes["total_avaliado"] = (
    detail_outcomes["Resolvida"] + detail_outcomes["Não Resolvida"]
)

detail_outcomes["nao_resolvidas_pct"] = (
    detail_outcomes["Não Resolvida"]
    .div(detail_outcomes["total_avaliado"].replace(0, float("nan")))
    * 100
)

detail_outcomes = detail_outcomes.reset_index()

# Evita destacar percentuais extremos baseados em poucas reclamações.
detail_outcomes = detail_outcomes[
    detail_outcomes["total_avaliado"] >= 30
].sort_values(
    ["nao_resolvidas_pct", "total_avaliado"],
    ascending=[False, False],
)

detail_outcomes.to_csv(
    GOLD_DIR / "company_desfecho_por_problema_2025.csv",
    index=False,
)

print("\n=== INDICADORES GERAIS ===")
print(summary.to_string(index=False))

print("\n=== GRUPOS COM MAIOR % DE NÃO RESOLVIDAS ===")
print(outcomes.head(10).to_string(index=False))

print("\n=== PROBLEMAS COM MAIOR % DE NÃO RESOLVIDAS ===")
print(detail_outcomes.head(15).to_string(index=False))

assert problems["quantidade"].sum() == total
assert len(avaliacoes_validas) == 19521
assert nao_resolvidas == 4606

# Evolução mensal.
company["mes"] = company["Data Abertura"].dt.to_period("M").astype(str)
monthly = (
    company.groupby("mes")
    .size()
    .reset_index(name="reclamacoes")
    .sort_values("mes")
)
monthly.to_csv(GOLD_DIR / "company_reclamacoes_mensais_2025.csv", index=False)

print("\n=== INDICADORES GERAIS ===")
print(summary.to_string(index=False))

print("\n=== TOP 10 GRUPOS DE PROBLEMAS ===")
for _, row in problems.head(10).iterrows():
    print(
        f"{row['Grupo Problema']}: "
        f"{int(row['quantidade']):,} reclamações "
        f"({row['percentual']:.2f}%)"
    )

print("\n=== TOP 10 PROBLEMAS DETALHADOS ===")
print(details.head(10).to_string(index=False))

print("\n=== RECLAMAÇÕES POR MÊS ===")
print(monthly.to_string(index=False))

print("\nArquivos gerados:")
print(SILVER_DIR / "company_reclamacoes_2025.csv")
for filename in [
    "company_indicadores_2025.csv",
    "company_grupos_problema_2025.csv",
    "company_problemas_detalhados_2025.csv",
    "company_reclamacoes_mensais_2025.csv",
    "company_desfecho_por_grupo_2025.csv",
    "company_desfecho_por_problema_2025.csv",
]:
    print(GOLD_DIR / filename)

print("\n=== VALIDAÇÃO ===")
print("Total de registros da company:", len(company))
print("\nSituação:")
print(company["Situação"].value_counts(dropna=False).to_string())

print("\nAvaliação da reclamação:")
print(
    company["Avaliação Reclamação"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nRespondida:")
print(company["Respondida"].value_counts(dropna=False).to_string())

assert problems["quantidade"].sum() == len(company), (
    "A soma dos grupos não corresponde ao total de registros."
)
