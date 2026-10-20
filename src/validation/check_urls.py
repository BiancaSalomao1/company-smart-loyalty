
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

    print(f"\nFonte: {source['name']}")

    try:
        response = requests.get(
            source["url"],
            timeout=20,
            headers={"User-Agent": "companySmartLoyalty/0.1"},
            allow_redirects=True,
            stream=True,
        )

        print(f"Status HTTP: {response.status_code}")
        print(f"URL final: {response.url}")
        print(
            "Content-Type:",
            response.headers.get("Content-Type", "não informado"),
        )
        response.close()

    except requests.RequestException as error:
        print(f"Erro de acesso: {error}")
