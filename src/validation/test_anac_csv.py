
from pathlib import Path
import csv

path = (
    Path(__file__).resolve().parents[2]
    / "data" / "bronze" / "anac"
    / "dadosconsumidor2025.csv"
)

# Exibe as primeiras linhas como texto bruto.
with path.open(
    "r", encoding="utf-8-sig", errors="replace", newline=""
) as file:
    for number in range(5):
        line = file.readline()
        print(f"Linha {number + 1}: {line[:1000]!r}")

print("\nTestando delimitadores...")

for delimiter in [";", ",", "\t", "|"]:
    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as file:
            reader = csv.reader(
                file,
                delimiter=delimiter,
                strict=False,
            )
            header = next(reader)

        print(
            f"Delimitador {delimiter!r}: "
            f"{len(header)} coluna(s)"
        )
        print("Cabeçalho:", header[:15])

    except Exception as error:
        print(f"Delimitador {delimiter!r}: erro: {error}")
