# Arquitetura

## Fluxo de dados

```text
Fontes públicas
      |
      v
Bronze — dados originais
      |
      v
Silver — dados tratados
      |
      v
Gold — indicadores e recomendações
      |
      v
DuckDB — banco analítico
      |
      v
Streamlit — dashboard
```

## Componentes

- **Ingestion:** download dos dados públicos.
- **Analysis:** transformação, indicadores, recomendações e simulação financeira.
- **Validation:** inspeção de fontes, URLs e formatos.
- **Analytics:** consolidação dos arquivos Gold em DuckDB.
- **Dashboard:** visualização interativa dos resultados.

## Evidência e simulação

Os indicadores de reclamações são derivados de dados públicos observados. Os cenários financeiros usam premissas hipotéticas e devem ser identificados como simulados.

A análise pública não permite afirmar que cada reclamação corresponde a um cliente único nem comprovar churn ou CLV individual.
