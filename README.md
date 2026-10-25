# ✈️ Radar de Alto Valor — Azul Fidelidade
### Copiloto Inteligente de CRM Preditivo, Explicabilidade SHAP e Retenção Human-in-the-Loop

> **Hacka Business · Desafio 3: Azul Linhas Aéreas**  
> **Time 37** · Outubro de 2026  
> **Repositório:** `company-smart-loyalty`

---

## 1. Resumo Executivo e Tese de Negócio

No setor aéreo brasileiro, adquirir um novo cliente custa entre **5 a 7 vezes mais** do que reter um passageiro frequente. Contudo, os sinais de atrito operacional (cancelamentos, atrasos de malha e extravio de bagagens) chegam fragmentados e reativos aos times de relacionamento, resultando em cancelamentos silenciosos de clientes de alto valor.

O **Radar de Alto Valor** é uma solução de inteligência de clientes que une **Machine Learning tabular**, **explicabilidade individual SHAP**, **Agente Cognitivo com RAG em regras regulamentares** e **governança Human-in-the-Loop**. A plataforma responde às 3 perguntas críticas da diretoria de CRM da Azul:

```
┌─────────────────────────┐       ┌─────────────────────────┐       ┌─────────────────────────┐
│     QUEM VALE MAIS?     │       │   QUEM SAI E POR QUÊ?   │       │      O QUE FAZER?       │
│  Score de Alto Valor e  │ ───>  │  Classificador de Churn │ ───>  │  Agente Cognitivo RAG   │
│   CLV Preditivo (12M)   │       │   Explicabilidade SHAP  │       │  Guardrails + HITL      │
└─────────────────────────┘       └─────────────────────────┘       └─────────────────────────┘
```

- **Proteção de Receita Ativa:** 76,17% da receita total anual concentra-se nos 10% superiores da base de clientes (Top Decil).
- **Detecção Antecipada:** Identificação da probabilidade de evasão até 4 meses antes da perda definitiva de engajamento.
- **Prescrição Sem Alucinações:** Ações de retenção rigorosamente amparadas pelo Regulamento Oficial do Programa Azul Fidelidade.

---

## 2. Métricas Técnicas e Resultados Validados (Fases 1, 2 e 3)

Todas as metas analíticas do MVP estipuladas no desafio foram superadas com folga, comprovadas via testes estatísticos e validação cruzada estratificada:

| Dimensão Analítica | Métrica do Projeto | Meta Mínima do Desafio | Resultado Obtido | Status |
|---|---|:---:|:---:|:---:|
| **Concentração de Receita (F2)** | % da Receita nos 10% Maiores CLVs | $\ge 60,00\%$ | **76,17%** | ✅ Superada (+16,17 p.p.) |
| **Poder de Separação de Churn (F3)** | ROC AUC (Classificador Tabular) | $\ge 0,7500$ | **0,9849** | ✅ Superada (+0,2349) |
| **Eficiência no Top Decil (F3)** | Lift de Churn no Top 10% | $\ge 2,00\times$ | **2,36x** | ✅ Superada (+0,36x) |
| **Acurácia Global do Modelo (F3)** | Accuracy / Precision / Recall | $\ge 80,00\%$ | **92,67% / 90,70% / 92,13%** | ✅ Superada |
| **Segmentação da Aviação (F4)** | Personas Acionáveis (K-Means) | 4 a 5 personas | **5 Personas Acionáveis** | ✅ Atingida (Silhouette 0,3289) |
| **Segurança do Agente RAG (F5)** | Benefícios Não Autorizados / Alucinações | 0 benefícios inventados | **0 Alucinações (100% auditado)** | ✅ Meta Zero Atingida |
| **Cobertura de Testes (F6)** | Testes Automatizados Unitários/E2E | $100\%$ passando | **28 de 28 testes OK** | ✅ 100% Verde |

### Distribuição das 5 Personas Acionáveis da Aviação (K-Means)
1. **Conectividade Regional (26,6%):** Voos frequentes interior-hub (alta margem média de rota ~26-28%).
2. **Ocasional em Ascensão (25,3%):** Clientes em início de relacionamento no nível Básico/Topázio.
3. **Alto Valor em Risco (19,9%):** Clientes com alto CLV pregresso e probabilidade de churn $\ge 50\%$.
4. **Lazer Premium (17,2%):** Alto ticket médio individual, viagens em família e alta sensibilidade a bagagem.
5. **Executivo Frequente (11,0%):** Membros Diamante de pontes aéreas corporativas (Espaço Azul, Lounge VCP).

---

## 3. Arquitetura em Camadas e Portabilidade Corporativa

A solução foi projetada no paradigma moderno de Data Mesh / Lakehouse em camadas medalhão, com schemas ANSI SQL 100% portáveis para **Snowflake**, **Databricks** ou **Google BigQuery**:

```
[ ANAC / Consumidor.gov.br 2025 ]      [ Catálogo da Malha Azul & Tiers ]
                 │                                      │
                 ▼                                      ▼
       CAMADA BRONZE (RAW)                     BASE SINTÉTICA CALIBRADA
   Hash MD5 & Validação de Schema           1.500 clientes · 11.178 voos
                 │                                      │
                 └──────────────────┬───────────────────┘
                                    ▼
                         CAMADA SILVER (CURATED)
                   Limpeza, Padronização e Deduplicação
                                    │
                                    ▼
                          CAMADA GOLD (ANALYTICS)
             Features RFM, Margem de Rota e Matriz de Churn
                                    │
                                    ▼
                     BANCO ANALÍTICO (DUCKDB / SQL)
                 `company_analytics.duckdb` (Zero Latency)
                                    │
                 ┌──────────────────┴──────────────────┐
                 ▼                                     ▼
     AGENTE COGNITIVO (RAG)                 COCKPIT STREAMLIT (HITL)
Base Vetorial / Lexical Regulamento        Matriz Valor x Risco, SHAP e
 Guardrails de Zero Alucinações            Exportação de Lotes (Salesforce)
```

- **Camada Bronze:** Ingestão validada com checagem de hash SHA-256 e schema das reclamações reais da ANAC (2025).
- **Camada Silver/Gold:** Transações e perfis sintéticos calibrados rigorosamente nas regras de negócio da Azul: 4 tiers oficiais (*Básico, Topázio, Safira, Diamante*) e 15 rotas estratégicas com suas respectivas margens operacionais (pontes aéreas, hubs VCP/CNF/REC e rotas regionais).
- **Camada de Serviço (DuckDB):** Banco colunar analítico embutido, dispensando infraestrutura pesada externa e garantindo consultas instantâneas no dashboard.

---

## 4. IA Responsável, LGPD e Explicabilidade SHAP

1. **Privacidade por Design (LGPD):** Nenhum dado pessoal identificável (PII) de passageiros reais foi utilizado ou exposto. A base transacional foi sintetizada probabilisticamente mantendo estrita fidelidade matemática às distribuições reais do setor aéreo.
2. **Explicabilidade SHAP Individual (Sem Caixas-Pretas):** Cada predição de churn possui decomposição aditiva local via valores de Shapley (`shap.TreeExplainer`). O analista de CRM sabe exatamente por que um cliente está em risco (ex.: *Recência elevada > 4 meses*, *Reclamação operacional pendente de bagagem*, *Queda de frequência em rotas nobres*).
3. **Guardrails de Zero Alucinações:** A esteira de pós-geração valida se qualquer benefício sugerido (pontos qualificáveis, franquia de bagagem, Espaço Azul, Lounge Viracopos ou upgrade) pertence legitimamente ao Tier do cliente no Regulamento Oficial do Azul Fidelidade.
4. **Governança Human-in-the-Loop (HITL):** A IA não dispara mensagens sozinhas. Todas as ações geradas recebem status *"Pendente de Avaliação Humana"* e exigem aprovação/revisão explícita do analista no dashboard antes de integrar o lote de disparo.

---

## 5. Resiliência Operacional e Plano B (Fallback Determinístico)

Para assegurar **100% de disponibilidade em qualquer ambiente de avaliação ou banca dos jurados**:

- O módulo do agente ([`src/agent/action_agent.py`](src/agent/action_agent.py)) detecta dinamicamente a presença de chaves de API (`OPENAI_API_KEY` ou `GEMINI_API_KEY`).
- Caso as chaves estejam ausentes ou ocorra indisponibilidade de rede/cota, o sistema chaveia **automaticamente e de forma transparente** para o **Motor Determinístico de Regras**.
- O fallback utiliza templates calibrados por *Persona × Tier × Fator de Risco SHAP*, recupera os trechos regulamentares pelo RAG e formata a resposta no schema JSON estrito de CRM, mantendo citação literal das seções do regulamento.

---

## 6. Estrutura do Repositório

```
company-smart-loyalty/
├── config/                     # Configurações de fontes de dados e parâmetros de ROI
├── data/
│   ├── analytics/              # Banco DuckDB (company_analytics.duckdb)
│   ├── bronze/                 # Dados brutos da ANAC validados
│   ├── documents/fidelidade/   # Regulamento oficial do Azul Fidelidade para o RAG
│   └── gold/                   # Perfis 360°, métricas e tabelas analíticas prontas
├── docs/                       # Documentação técnica, arquitetura e roteiro do pitch
│   ├── regras_azul_fidelidade.md
│   ├── roteiro-pitch.md
│   └── synthetic-data.md
├── src/
│   ├── agent/                  # Agente Cognitivo de CRM com RAG e Guardrails (F5)
│   │   ├── __init__.py
│   │   └── action_agent.py
│   ├── analysis/               # Pipelines de CLV, Churn LightGBM, SHAP e DuckDB
│   │   ├── build_analytics_db.py
│   │   ├── build_customer_intelligence.py
│   │   └── process_anac_year.py
│   ├── dashboard/              # Interface Web Interativa (Streamlit)
│   │   └── app.py
│   ├── ingestion/              # Ingestão e download de fontes da ANAC
│   ├── synthetic/              # Gerador calibrado de passageiros e rotas Azul
│   └── validation/             # Validação estrutural de arquivos e dados
├── tests/                      # Bateria de 28 testes automatizados (unittest / AppTest)
│   ├── test_action_agent.py
│   ├── test_crm_cockpit_flow.py
│   └── test_dashboard.py
├── test_action_agent.py        # Script demonstrativo standalone para jurados
├── Dockerfile                  # Empacotamento de contêiner para produção
├── compose.yaml                # Orquestração simplificada com Docker Compose
└── requirements.txt            # Dependências Python fixadas
```

---

## 7. Guia Rápido de Execução

### Pré-requisitos
- Python 3.10, 3.11 ou 3.12 (compatível com Windows, Linux e macOS)
- Docker Desktop (opcional para execução conteinerizada)

### Opção A: Execução Local Rápida

1. **Clonar e criar ambiente virtual:**
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd company-smart-loyalty
   python -m venv .venv
   
   # No Windows:
   .venv\Scripts\activate
   # No Linux/macOS:
   source .venv/bin/activate
   ```

2. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Gerar a base sintética e o banco analítico DuckDB:**
   ```bash
   python src/synthetic/generate_passengers.py
   python src/analysis/build_customer_intelligence.py
   python src/analysis/build_analytics_db.py
   ```

4. **Rodar os testes de conformidade (Garantia de 100% OK):**
   ```bash
   python -m unittest discover tests
   ```

5. **Iniciar o Dashboard Executivo:**
   ```bash
   streamlit run src/dashboard/app.py
   ```
   Acesse no navegador: **`http://localhost:8501`**.

---

### Opção B: Execução via Docker Compose (1 Comando)

Caso prefira rodar sem instalar dependências no host:

```bash
# Subir o contêiner em segundo plano
docker compose up -d --build
```

- Acesse: **`http://localhost:8501`**
- Para encerrar: `docker compose down`

---

### Opção C: Teste Standalone do Agente Cognitivo (Linha de Comando)

Para avaliar o comportamento do Agente de CRM e dos Guardrails sem abrir o navegador:

```bash
python test_action_agent.py
```
*O script simula passageiros de diferentes personas (incluindo clientes Safira e Diamante em risco), exibindo o payload JSON gerado e a auditoria de zero benefícios alucinados.*

---

## 8. Licença e Créditos

Desenvolvido pelo **Time 37** para o **Hacka Business — Desafio 3 (Azul Linhas Aéreas)**.  
Fontes de dados públicos: **ANAC / Consumidor.gov.br** (Ano base 2025).  
Regulamento: **Programa Azul Fidelidade** (Políticas e Categorias Públicas Oficiais).
