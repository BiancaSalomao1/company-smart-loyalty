
from pathlib import Path

import duckdb


# Raiz do projeto: company-smart-loyalty/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

GOLD_DIR = PROJECT_ROOT / "data" / "gold"
DB_DIR = PROJECT_ROOT / "data" / "analytics"
DB_PATH = DB_DIR / "company_analytics.duckdb"


# Arquivos Gold que serão disponibilizados para consultas analíticas.
GOLD_FILES = {
    "indicadores_2025": "company_indicadores_2025.csv",
    "grupos_problema_2025": "company_grupos_problema_2025.csv",
    "problemas_detalhados_2025": "company_problemas_detalhados_2025.csv",
    "reclamacoes_mensais_2025": "company_reclamacoes_mensais_2025.csv",
    "desfecho_por_grupo_2025": "company_desfecho_por_grupo_2025.csv",
    "desfecho_por_problema_2025": "company_desfecho_por_problema_2025.csv",
    "recomendacoes_2025": "company_recomendacoes_2025.csv",
    "roi_scenarios": "roi_scenarios.csv",
    "perfis_clientes_sinteticos": "perfis_clientes_sinteticos.csv",
    "dim_rotas_azul": "dim_rotas_azul.csv",
    "fact_transacoes_sinteticas": "fact_transacoes_sinteticas.csv",
    "influenciadores_viagem": "influenciadores_viagem.csv",
}

YEARLY_FILE_PREFIXES = {
    "indicadores": "company_indicadores_",
    "desfecho_por_problema": "company_desfecho_por_problema_",
    "recomendacoes": "company_recomendacoes_",
    "reclamacoes_mensais": "company_reclamacoes_mensais_",
}


def yearly_gold_files() -> dict[str, Path]:
    """Descobre saídas anuais sem apagar ou mesclar anos anteriores."""
    files = {}
    for table_prefix, file_prefix in YEARLY_FILE_PREFIXES.items():
        for path in GOLD_DIR.glob(f"{file_prefix}[0-9][0-9][0-9][0-9].csv"):
            year = path.stem.rsplit("_", 1)[-1]
            files[f"{table_prefix}_{year}"] = path
    return files


def create_database() -> None:
    """Cria o banco DuckDB e importa os arquivos CSV da camada Gold."""

    DB_DIR.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(str(DB_PATH))

    try:
        print("=== CONSTRUINDO BANCO ANALÍTICO ===")
        print(f"Banco: {DB_PATH}\n")

        imported = 0

        for table_name, filename in GOLD_FILES.items():
            csv_path = GOLD_DIR / filename

            if not csv_path.exists():
                print(f"[AVISO] Arquivo não encontrado: {filename}")
                continue

            # read_csv_auto identifica delimitadores, cabeçalhos
            # e tipos de dados automaticamente.
            csv_path_sql = str(csv_path).replace("'", "''")

            connection.execute(
                f"""
                CREATE OR REPLACE TABLE "{table_name}" AS
                SELECT *
                FROM read_csv_auto(
                    '{csv_path_sql}',
                    header = true,
                    sample_size = -1,
                    nullstr = ''
                )
                """
            )

            row_count = connection.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]

            print(
                f"[OK] {table_name}: "
                f"{row_count:,} registros"
            )

            imported += 1

        for table_name, csv_path in yearly_gold_files().items():
            csv_path_sql = str(csv_path).replace("'", "''")
            connection.execute(
                f'''CREATE OR REPLACE TABLE "{table_name}" AS
                SELECT * FROM read_csv_auto('{csv_path_sql}', header = true, sample_size = -1, nullstr = '')'''
            )
            row_count = connection.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            print(f"[OK] {table_name}: {row_count:,} registros")
            imported += 1

        print(f"\nTabelas importadas: {imported}")

        # Visão consolidada dos cenários financeiros.
        # Os valores continuam sendo hipóteses simuladas.
        connection.execute(
            """
            CREATE OR REPLACE VIEW vw_roi_scenarios AS
            SELECT
                scenario_name,
                investment_brl,
                gross_incremental_benefit_brl,
                net_benefit_brl,
                roi_pct,
                payback_months,
                'SIMULADO' AS tipo_resultado
            FROM roi_scenarios
            """
        ) if table_exists(connection, "roi_scenarios") else None

        # Visão mestre analítica de inteligência de clientes
        connection.execute(
            """
            CREATE OR REPLACE VIEW customer_intelligence AS
            SELECT * FROM perfis_clientes_sinteticos
            """
        ) if table_exists(connection, "perfis_clientes_sinteticos") else None

        connection.execute(
            """
            CREATE OR REPLACE VIEW vw_customer_360 AS
            SELECT
                customer_id,
                loyalty_tier_synthetic AS tier_atual,
                months_since_last_flight AS recencia_dias,
                flights_last_12m AS frequencia_12m,
                total_spend_12m_brl AS valor_total_12m,
                share_rotas_nobres,
                service_issues_last_12m AS reclamacoes_relacionadas,
                clv_simulado_12m_brl AS clv_score,
                churn_risk_score AS churn_prob,
                top_shap_reason,
                persona
            FROM perfis_clientes_sinteticos
            """
        ) if table_exists(connection, "perfis_clientes_sinteticos") else None

        print("\n=== TABELAS DISPONÍVEIS ===")

        tables = connection.execute(
            "SHOW TABLES"
        ).fetchall()

        for (table_name,) in tables:
            print(f"- {table_name}")

        print("\nBanco analítico criado com sucesso!")

    finally:
        connection.close()


def table_exists(connection, table_name: str) -> bool:
    """Verifica se uma tabela existe no banco."""
    result = connection.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
        """,
        [table_name],
    ).fetchone()

    return result[0] > 0


if __name__ == "__main__":
    create_database()
