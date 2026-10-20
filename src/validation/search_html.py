
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "sources.yaml"

TERMS = {
    "anac_tarifas": [
        "tarifa",
        "passageiros",
        "company",
        "dados abertos",
        "download",
    ],
    "reclame_aqui": [
        "reclamação",
        "não resolvido",
        "voltaria a fazer negócio",
        "nota",
        "company",
    ],
}

with CONFIG_PATH.open(encoding="utf-8") as file:
    config = yaml.safe_load(file)

for source in config["sources"]:
    name = source["name"]

    if not source.get("enabled", False):
        continue

    if name == "company_fidelidade":
        print(f"\n{name}: acesso automatizado bloqueado anteriormente.")
        continue

    print(f"\n{'=' * 60}\nFonte: {name}")

    try:
        response = requests.get(
            source["url"],
            timeout=20,
            headers={"User-Agent": "companySmartLoyalty/0.1"},
        )
        print(f"Status HTTP: {response.status_code}")

        if response.status_code != 200:
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove elementos que normalmente não representam conteúdo editorial.
        for element in soup(["script", "style", "noscript", "svg"]):
            element.decompose()

        text = " ".join(soup.stripped_strings).lower()

        for term in TERMS.get(name, []):
            print(f"'{term}': {text.count(term)} ocorrência(s)")

        print("\nTítulo:", soup.title.get_text(strip=True) if soup.title else "Não encontrado")
        print("Links encontrados:", len(soup.find_all("a", href=True)))

    except requests.RequestException as error:
        print(f"Erro de acesso: {error}")
