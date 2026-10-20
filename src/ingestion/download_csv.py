from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import requests

BRONZE = Path("data/bronze")
BRONZE.mkdir(parents=True, exist_ok=True)


def download_csv(name: str, url: str) -> Path:
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    content = response.content
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    directory = BRONZE / name
    directory.mkdir(parents=True, exist_ok=True)

    file_path = directory / f"{timestamp}.csv"
    file_path.write_bytes(content)

    metadata = {
        "source": name,
        "url": url,
        "collected_at": timestamp,
        "sha256": hashlib.sha256(content).hexdigest(),
        "status_code": response.status_code,
        "file": str(file_path),
    }

    metadata_path = file_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return file_path


if __name__ == "__main__":
    # Substituir pela URL direta de um CSV público validado.
    url = "URL_DIRETA_DO_CSV"
    path = download_csv("anac_tarifas", url)
    print(f"Arquivo salvo em: {path}")