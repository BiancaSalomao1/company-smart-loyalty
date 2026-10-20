
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]

URL = (
    "https://sistemas.anac.gov.br/dadosabertos/"
    "Voos%20e%20opera%C3%A7%C3%B5es%20a%C3%A9reas/"
    "Dados%20do%20consumidor.gov/"
    "dadosconsumidor2025.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "bronze" / "anac"
OUTPUT_FILE = OUTPUT_DIR / "dadosconsumidor2025.csv"
METADATA_FILE = OUTPUT_DIR / "dadosconsumidor2025_metadata.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Baixando arquivo da ANAC...")

response = requests.get(
    URL,
    timeout=120,
    headers={"User-Agent": "companySmartLoyalty/0.1"},
)
response.raise_for_status()

content_type = response.headers.get("Content-Type", "")
if "text/html" in content_type.lower():
    raise RuntimeError(
        "O servidor retornou HTML em vez do CSV. "
        "Verifique a URL antes de continuar."
    )

content = response.content

if not content:
    raise RuntimeError("O arquivo recebido está vazio.")

OUTPUT_FILE.write_bytes(content)

sha256 = hashlib.sha256(content).hexdigest()

metadata = {
    "source": "ANAC - Dados do Consumidor.gov.br",
    "year": 2025,
    "url": URL,
    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    "file": OUTPUT_FILE.name,
    "size_bytes": len(content),
    "sha256": sha256,
    "content_type": content_type,
}

METADATA_FILE.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Arquivo salvo: {OUTPUT_FILE}")
print(f"Tamanho: {len(content):,} bytes")
print(f"SHA-256: {sha256}")


# O arquivo tem uma linha de atualização antes do cabeçalho.
# O separador é ; e os campos estão entre aspas duplas.
df = pd.read_csv(
    OUTPUT_FILE,
    sep=";",
    skiprows=1,
    quotechar='"',
    encoding="utf-8-sig",
    dtype=str,
    na_values=["null"],
    keep_default_na=False,
)

print(f"\nTotal de registros: {len(df):,}")
print(f"Total de colunas: {len(df.columns)}")

print("\nColunas identificadas:")
for column in df.columns:
    print(f"- {column}")

print("\nPrimeiros cinco registros:")
print(df.head().to_string(index=False))

print("\nEmpresas mais frequentes:")
print(df["Nome Fantasia"].value_counts().head(10).to_string())
