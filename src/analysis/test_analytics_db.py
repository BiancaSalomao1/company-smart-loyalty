
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "analytics" / "company_analytics.duckdb"

connection = duckdb.connect(str(DB_PATH), read_only=True)

try:
    print("\n=== CENÁRIOS DE ROI ===")

    result = connection.execute(
        """
        SELECT
            scenario_name,
            investment_brl,
            net_benefit_brl,
            roi_pct,
            tipo_resultado
        FROM vw_roi_scenarios
        ORDER BY roi_pct DESC
        """
    ).df()

    print(result.to_string(index=False))

    print("\n=== PROBLEMAS COM MAIOR TAXA DE NÃO RESOLUÇÃO ===")

    result = connection.execute(
        """
        SELECT
            Problema,
            total_avaliado,
            nao_resolvidas_pct
        FROM desfecho_por_problema_2025
        WHERE total_avaliado >= 30
        ORDER BY nao_resolvidas_pct DESC
        LIMIT 10
        """
    ).df()

    print(result.to_string(index=False))

finally:
    connection.close()
