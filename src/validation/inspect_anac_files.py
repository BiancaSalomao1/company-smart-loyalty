
from urllib.parse import urljoin
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URL = (
    "https://sistemas.anac.gov.br/dadosabertos/"
    "Voos%20e%20opera%C3%A7%C3%B5es%20a%C3%A9reas/"
    "Dados%20do%20consumidor.gov/"
)

response = requests.get(
    URL,
    timeout=30,
    headers={"User-Agent": "companySmartLoyalty/0.1"},
)

print("Status HTTP:", response.status_code)
print("URL final:", response.url)
print("Content-Type:", response.headers.get("Content-Type"))

response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

print("\nArquivos e links encontrados:")

for link in soup.find_all("a", href=True):
    href = link["href"]
    label = link.get_text(" ", strip=True)
    full_url = urljoin(response.url, href)

    if any(ext in href.lower() for ext in (
        ".csv", ".json", ".zip", ".xlsx", ".xls", ".txt"
    )):
        print(f"{label or href}\n  {full_url}")
