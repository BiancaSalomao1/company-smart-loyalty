
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "sources.yaml"

with CONFIG_PATH.open(encoding="utf-8") as file:
    config = yaml.safe_load(file)

for source in config["sources"]:
    if not source.get("enabled", False):
        continue

    print(f"\n{'=' * 60}")
    print(f"Fonte: {source['name']}")

    try:
        response = requests.get(
            source["url"],
            timeout=20,
            headers={"User-Agent": "companySmartLoyalty/0.1"},
        )

        print(f"Status HTTP: {response.status_code}")

        if response.status_code != 200:
            print("Conteúdo não inspecionado por causa do status HTTP.")
            continue

        html = response.text
        print(f"Tamanho do HTML: {len(html)} caracteres")
        print("\nPrévia do HTML:")
        print(html[:800])

    except requests.RequestException as error:
        print(f"Erro: {error}")
