
from pathlib import Path
import yaml

config_path = Path("config/sources.yaml")

with config_path.open(encoding="utf-8") as file:
    config = yaml.safe_load(file)

sources = config.get("sources", [])

for source in sources:
    required = {"name", "url", "format", "enabled"}
    missing = required - source.keys()

    if missing:
        print(f"Fonte inválida: {source.get('name', 'sem nome')}")
        print(f"Campos ausentes: {missing}")
        continue

    print(
        f"{source['name']}: "
        f"format={source['format']}, "
        f"enabled={source['enabled']}"
    )
