# Company Smart Loyalty

Plataforma de análise de experiência do cliente e apoio à priorização de ações de fidelização.

## Objetivo

Utilizar dados públicos de reclamações para identificar padrões de problemas, gerar recomendações de atuação e explorar cenários financeiros hipotéticos relacionados à retenção.

## Funcionalidades

- Ingestão de dados públicos da ANAC / Consumidor.gov.br.
- Tratamento e análise de reclamações.
- Indicadores por mês, grupo e tipo de problema.
- Recomendações baseadas em regras.
- Simulação de ROI com premissas configuráveis.
- Armazenamento analítico em DuckDB.
- Dashboard interativo com Streamlit.
- Execução conteinerizada com Docker Compose.

## Tecnologias

Python, Pandas, DuckDB, Streamlit, Plotly, Docker e Docker Compose.

## Estrutura

- `config/`: configurações das fontes e cenários.
- `data/`: dados locais e resultados gerados.
- `docs/`: documentação técnica.
- `sql/`: consultas SQL.
- `src/analysis/`: análises, indicadores e recomendações.
- `src/dashboard/`: dashboard Streamlit.
- `src/ingestion/`: ingestão de dados.
- `src/validation/`: validação das fontes.

## Dados e limitações

A análise utiliza dados públicos de reclamações referentes a 2025. Reclamações não representam necessariamente clientes únicos e não comprovam churn.

Os cenários de ROI são simulados com premissas configuráveis e não representam resultados financeiros observados.

## Execução local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Após preparar os dados e gerar o banco analítico:

```bash
streamlit run src/dashboard/app.py
```

## Execução com Docker

Após preparar os dados e o banco DuckDB:

```bash
docker compose up -d --build
```

Acesse http://localhost:8501.

Para encerrar:

```bash
docker compose down
```

## Status

Protótipo em desenvolvimento para um hackathon. A integração com dados privados de clientes e transações ainda não está implementada.
