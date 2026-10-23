"""Transforma um CSV anual validado da ANAC em indicadores e recomendações."""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze" / "anac"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
REQUIRED_COLUMNS = {
    "Nome Fantasia", "Data Abertura", "Grupo Problema", "Problema",
    "Respondida", "Avaliação Reclamação", "Nota do Consumidor",
    "Tempo Resposta (em dias)",
}


def action_for_problem(problem: str) -> str:
    value = str(problem).casefold()
    if any(word in value for word in ["atraso", "cancelamento", "embarque"]):
        return "comunicação e acompanhamento operacional"
    if any(word in value for word in ["bagagem", "extravio", "avaria"]):
        return "acompanhamento de bagagem"
    if any(word in value for word in ["cobrança", "reembolso", "tarifa", "valor"]):
        return "resolução financeira"
    return "análise e tratamento direcionado"


def process_year(year: int, company_name: str, input_file: Path | None = None, output_dir: Path = GOLD_DIR) -> dict:
    """Gera arquivos Gold separados por ano a partir de um CSV já validado."""
    source = input_file or BRONZE_DIR / f"dadosconsumidor{year}.csv"
    frame = pd.read_csv(source, sep=";", skiprows=1, quotechar='"', encoding="utf-8-sig", dtype=str)
    frame.columns = frame.columns.str.strip()
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")

    frame["Data Abertura"] = pd.to_datetime(frame["Data Abertura"], errors="coerce")
    company = frame[
        frame["Nome Fantasia"].str.casefold().eq(company_name.casefold())
        & frame["Data Abertura"].dt.year.eq(year)
    ].copy()
    if company.empty:
        raise ValueError(f"Nenhuma reclamação encontrada para {company_name} em {year}.")

    company["Nota do Consumidor"] = pd.to_numeric(company["Nota do Consumidor"], errors="coerce")
    company["Tempo Resposta (em dias)"] = pd.to_numeric(company["Tempo Resposta (em dias)"], errors="coerce")
    evaluated = company[company["Avaliação Reclamação"].isin(["Resolvida", "Não Resolvida"])]
    unresolved = int((evaluated["Avaliação Reclamação"] == "Não Resolvida").sum())
    summary = pd.DataFrame([{
        "ano": year, "empresa": company_name, "total_reclamacoes": len(company),
        "reclamacoes_respondidas_pct": (company["Respondida"] == "S").mean() * 100,
        "reclamacoes_avaliadas": len(evaluated),
        "reclamacoes_resolvidas": int((evaluated["Avaliação Reclamação"] == "Resolvida").sum()),
        "reclamacoes_nao_resolvidas": unresolved,
        "nao_resolvidas_pct_sobre_avaliadas": unresolved / len(evaluated) * 100 if len(evaluated) else None,
        "nota_media_consumidor": company["Nota do Consumidor"].mean(),
        "tempo_medio_resposta_dias": company["Tempo Resposta (em dias)"].mean(),
    }])
    outcomes = (
        evaluated.groupby(["Grupo Problema", "Problema", "Avaliação Reclamação"]).size()
        .unstack(fill_value=0).reset_index()
    )
    for column in ["Resolvida", "Não Resolvida"]:
        if column not in outcomes:
            outcomes[column] = 0
    outcomes["total_avaliado"] = outcomes["Resolvida"] + outcomes["Não Resolvida"]
    outcomes["nao_resolvidas_pct"] = outcomes["Não Resolvida"] / outcomes["total_avaliado"].replace(0, float("nan")) * 100
    recommendations = outcomes[outcomes["total_avaliado"] >= 30].copy()
    recommendations["acao_sugerida"] = recommendations["Problema"].map(action_for_problem)
    recommendations["tipo_evidencia"] = "dado_publico_observado"
    monthly = company.groupby(company["Data Abertura"].dt.month_name(locale="pt_BR.UTF-8") if False else company["Data Abertura"].dt.strftime("%Y-%m")).size().reset_index(name="reclamacoes")
    monthly.columns = ["mes", "reclamacoes"]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / f"company_indicadores_{year}.csv", index=False)
    outcomes.to_csv(output_dir / f"company_desfecho_por_problema_{year}.csv", index=False)
    recommendations.to_csv(output_dir / f"company_recomendacoes_{year}.csv", index=False)
    monthly.to_csv(output_dir / f"company_reclamacoes_mensais_{year}.csv", index=False)
    return {"year": year, "company": company_name, "records": len(company)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Processa uma base anual ANAC validada.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--company", required=True)
    args = parser.parse_args()
    print(process_year(args.year, args.company))


if __name__ == "__main__":
    main()
