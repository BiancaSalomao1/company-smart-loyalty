"""Baixa e valida um CSV anual da ANAC antes de promovê-lo à camada Bronze."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze" / "anac"
REQUIRED_COLUMNS = {
    "Nome Fantasia",
    "Data Abertura",
    "Grupo Problema",
    "Problema",
    "Avaliação Reclamação",
}


def default_url(year: int) -> str:
    return (
        "https://sistemas.anac.gov.br/dadosabertos/"
        "Voos%20e%20opera%C3%A7%C3%B5es%20a%C3%A9reas/"
        "Dados%20do%20consumidor.gov/"
        f"dadosconsumidor{year}.csv"
    )


def validate_csv(content: bytes, year: int) -> pd.DataFrame:
    """Valida estrutura e período antes de alterar arquivos persistidos."""
    if not content:
        raise ValueError("O arquivo recebido está vazio.")
    if len(content) < 100:
        raise ValueError("O arquivo recebido é pequeno demais para ser um CSV válido.")

    frame = pd.read_csv(
        BytesIO(content), sep=";", skiprows=1, quotechar='"',
        encoding="utf-8-sig", dtype=str, na_values=["null"],
    )
    frame.columns = frame.columns.str.strip()
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")

    dates = pd.to_datetime(frame["Data Abertura"], errors="coerce")
    if not (dates.dt.year == year).any():
        raise ValueError(f"O arquivo não contém registros do ano solicitado: {year}.")
    return frame


def update_dataset(year: int, url: str | None = None, output_dir: Path = BRONZE_DIR) -> dict:
    """Faz download, valida em memória e só então grava CSV e metadados."""
    if year < 2014 or year > datetime.now().year:
        raise ValueError("Ano fora do intervalo permitido para atualização.")

    source_url = url or default_url(year)
    response = requests.get(
        source_url, timeout=120, headers={"User-Agent": "companySmartLoyalty/0.2"}
    )
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "text/html" in content_type.lower():
        raise ValueError("A fonte retornou HTML em vez de um CSV.")

    frame = validate_csv(response.content, year)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"dadosconsumidor{year}.csv"
    metadata_path = output_dir / f"dadosconsumidor{year}_metadata.json"
    csv_path.write_bytes(response.content)

    metadata = {
        "source": "ANAC - Dados do Consumidor.gov.br",
        "year": year,
        "url": source_url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "file": csv_path.name,
        "rows_validated": len(frame),
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "content_type": content_type,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Atualiza a camada Bronze com validação.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--url", help="URL direta validada; omita para usar o padrão ANAC.")
    args = parser.parse_args()
    result = update_dataset(args.year, args.url)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
