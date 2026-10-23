
import json
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from src.knowledge.answer_question import answer_question
from src.ingestion.update_anac_dataset import update_dataset
from src.analysis.process_anac_year import process_year
from src.analysis.build_analytics_db import create_database


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "analytics" / "company_analytics.duckdb"
SYNTHETIC_METRICS_PATH = PROJECT_ROOT / "data" / "gold" / "metricas_modelo_sintetico.json"

st.set_page_config(
    page_title="Radar de Alto Valor | Company",
    page_icon="✈️",
    layout="wide",
)

# Paleta visual
AZUL = "#00529B"
AZUL_CLARO = "#EAF3FB"
VERDE = "#16865B"
VERMELHO = "#C0392B"
CINZA = "#667085"


# ============================================================
# ACESSO AOS DADOS
# ============================================================

@st.cache_data(ttl=300)
def read_query(query: str) -> pd.DataFrame:
    """Executa uma consulta somente leitura no DuckDB."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Banco não encontrado: {DB_PATH}. "
            "Execute primeiro build_analytics_db.py."
        )

    connection = duckdb.connect(str(DB_PATH), read_only=True)

    try:
        return connection.execute(query).df()
    finally:
        connection.close()


def money(value) -> str:
    """Formata valores em reais."""
    if pd.isna(value):
        return "—"

    formatted = f"{value:,.2f}"
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {formatted}"


def percentage(value) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.2f}%".replace(".", ",")


# ============================================================
# CABEÇALHO
# ============================================================

st.title("✈️ Radar de Alto Valor")
st.subheader("Inteligência de experiência do cliente e oportunidades de retenção")

st.markdown(
    """
    Transformando padrões de reclamações em prioridades de atuação
    e cenários de decisão para programas de fidelidade.
    """
)

st.info(
    "Fonte pública: ANAC / Consumidor.gov.br — ano de 2025. "
    "Reclamações não representam necessariamente clientes únicos nem comprovam churn. "
    "Os cenários de ROI são simulações baseadas em premissas configuráveis."
)

if not DB_PATH.exists():
    st.error(
        "Banco analítico não encontrado. Execute "
        "`python src/analysis/build_analytics_db.py` antes de abrir o dashboard."
    )
    st.stop()


# ============================================================
# CARREGAMENTO DOS INDICADORES
# ============================================================

try:
    indicadores = read_query(
        """
        SELECT *
        FROM indicadores_2025
        LIMIT 1
        """
    )

    mensal = read_query(
        """
        SELECT mes, reclamacoes
        FROM reclamacoes_mensais_2025
        """
    )

    problemas = read_query(
        """
        SELECT
            Problema,
            total_avaliado,
            nao_resolvidas_pct
        FROM desfecho_por_problema_2025
        WHERE total_avaliado >= 30
        ORDER BY nao_resolvidas_pct DESC
        """
    )

    grupos = read_query(
        """
        SELECT *
        FROM desfecho_por_grupo_2025
        """
    )

    roi = read_query(
        """
        SELECT
            scenario_id,
            scenario_name,
            data_type,
            eligible_units_assumption,
            action_cost_per_unit_brl,
            fixed_campaign_cost_brl,
            assumed_incremental_retention_pct,
            contribution_margin_per_retained_customer_brl,
            benefit_horizon_months,
            investment_brl,
            expected_retained_units_simulated,
            gross_incremental_benefit_brl,
            net_benefit_brl,
            roi_pct,
            payback_months,
            retention_validated
        FROM roi_scenarios
        ORDER BY roi_pct DESC
        """
    )

    recomendacoes = read_query(
        """
        SELECT *
        FROM recomendacoes_2025
        """
    )

except Exception as exc:
    st.error(f"Não foi possível carregar os dados: {exc}")
    st.stop()

try:
    customer_profiles = read_query(
        """
        SELECT customer_id, flights_last_12m, flights_previous_12m, months_since_last_flight,
               avg_ticket_brl, points_balance, service_issues_last_12m,
               engagement_score, churn_risk_score, clv_simulado_12m_brl,
               loyalty_tier_synthetic, preferred_route_synthetic, segment,
               recommended_action, recommendation_rationale, risk_explanation,
               suggested_channel, draft_message, estimated_action_cost_brl
        FROM perfis_clientes_sinteticos
        ORDER BY churn_risk_score DESC, clv_simulado_12m_brl DESC
        """
    )
except Exception:
    customer_profiles = pd.DataFrame()

try:
    influenciadores = read_query(
        """
        SELECT ranking, creator_name, focus_theme, primary_platform,
               subscriber_tier, content_alignment_score, loyalty_focus,
               avg_views_per_video, estimated_cpm_brl, fit_score,
               suggested_partnership, notes
        FROM influenciadores_viagem
        ORDER BY ranking ASC
        """
    )
except Exception:
    influenciadores = pd.DataFrame()


if indicadores.empty:
    st.warning("A tabela de indicadores está vazia.")
    st.stop()


# ============================================================
# FILTROS
# ============================================================

st.sidebar.header("Filtros")

st.sidebar.divider()
st.sidebar.subheader("Atualização anual da fonte pública")
selected_data_year = st.sidebar.number_input(
    "Ano dos dados ANAC",
    min_value=2014,
    max_value=datetime.now().year,
    value=2025,
    step=1,
    help="A base atual só é substituída após validação de estrutura e período.",
)
if st.sidebar.button("Validar e baixar nova base"):
    try:
        with st.spinner("Baixando e validando a fonte pública..."):
            update_metadata = update_dataset(int(selected_data_year))
    except Exception as exc:
        st.sidebar.error(f"A atualização não foi promovida: {exc}")
    else:
        st.sidebar.success(
            f"Base {update_metadata['year']} validada: "
            f"{update_metadata['rows_validated']:,} registros."
        )
        st.sidebar.info(
            "A etapa Bronze foi atualizada. Execute o processamento analítico "
            "do ano validado antes de trocar os indicadores exibidos."
        )

company_for_processing = st.sidebar.text_input(
    "Nome da empresa na fonte", value="Azul Linhas Aéreas",
    help="Deve corresponder exatamente ao campo 'Nome Fantasia' no CSV validado.",
)
if st.sidebar.button("Processar base validada e reconstruir banco"):
    try:
        with st.spinner("Gerando indicadores e reconstruindo o banco analítico..."):
            result = process_year(
                int(selected_data_year), company_for_processing
            )
            create_database()
            st.cache_data.clear()
    except Exception as exc:
        st.sidebar.error(f"O processamento não foi concluído: {exc}")
    else:
        st.sidebar.success(
            f"Ano {result['year']} processado: {result['records']:,} reclamações."
        )
        st.sidebar.info(
            "A tabela anual foi criada no DuckDB. A seleção de visualização por ano "
            "será aplicada em seguida; o painel atual permanece na referência de 2025."
        )

if not mensal.empty:
    meses_disponiveis = mensal["mes"].astype(str).tolist()
    meses_selecionados = st.sidebar.multiselect(
        "Meses para análise",
        options=meses_disponiveis,
        default=meses_disponiveis,
    )
    mensal_filtrado = mensal[
        mensal["mes"].astype(str).isin(meses_selecionados)
    ].copy()
else:
    mensal_filtrado = mensal.copy()

volume_minimo = st.sidebar.number_input(
    "Mínimo de avaliações por problema",
    min_value=1,
    max_value=10000,
    value=30,
    step=10,
)

problemas_filtrados = problemas[
    problemas["total_avaliado"] >= volume_minimo
].copy()

st.sidebar.caption("Os filtros afetam os gráficos e tabelas correspondentes.")


# ============================================================
# INDICADORES EXECUTIVOS
# ============================================================

dados = indicadores.iloc[0]

st.header("1. Visão executiva")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Reclamações",
    f"{int(dados['total_reclamacoes']):,}".replace(",", "."),
)

col2.metric(
    "Respondidas",
    percentage(dados["reclamacoes_respondidas_pct"]),
)

col3.metric(
    "Não resolução",
    percentage(dados["nao_resolvidas_pct_sobre_avaliadas"]),
    help="Percentual entre reclamações avaliadas como resolvidas ou não resolvidas.",
)

col4.metric(
    "Resposta média",
    f"{dados['tempo_medio_resposta_dias']:.1f} dias",
)

col5.metric(
    "Nota média",
    f"{dados['nota_media_consumidor']:.2f}/5",
    help="Média das notas disponíveis no conjunto analisado.",
)

st.caption(
    f"Avaliações válidas: {int(dados['reclamacoes_avaliadas']):,}. "
    f"Resolvidas: {int(dados['reclamacoes_resolvidas']):,}. "
    f"Não resolvidas: {int(dados['reclamacoes_nao_resolvidas']):,}."
    .replace(",", ".")
)


# ============================================================
# EVOLUÇÃO MENSAL
# ============================================================

st.header("2. Evolução mensal das reclamações")

if mensal_filtrado.empty:
    st.warning("Não há meses selecionados para exibir.")
else:
    ordem_meses = {
        "Janeiro": 1,
        "Fevereiro": 2,
        "Março": 3,
        "Abril": 4,
        "Maio": 5,
        "Junho": 6,
        "Julho": 7,
        "Agosto": 8,
        "Setembro": 9,
        "Outubro": 10,
        "Novembro": 11,
        "Dezembro": 12,
    }

    mensal_filtrado["ordem_mes"] = (
        mensal_filtrado["mes"].map(ordem_meses).fillna(99)
    )
    mensal_filtrado = mensal_filtrado.sort_values("ordem_mes")

    fig_mensal = px.line(
        mensal_filtrado,
        x="mes",
        y="reclamacoes",
        markers=True,
        title="Volume mensal de reclamações — 2025",
        labels={
            "mes": "Mês",
            "reclamacoes": "Reclamações",
        },
    )

    fig_mensal.update_traces(
        line=dict(color=AZUL, width=3),
        marker=dict(size=8),
    )
    fig_mensal.update_layout(
        xaxis_title="Mês",
        yaxis_title="Quantidade de reclamações",
        hovermode="x unified",
    )

    st.plotly_chart(fig_mensal, use_container_width=True)

    st.dataframe(
        mensal_filtrado[["mes", "reclamacoes"]].rename(
            columns={
                "mes": "Mês",
                "reclamacoes": "Reclamações",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PRIORIZAÇÃO DE PROBLEMAS
# ============================================================

st.header("3. Radar de problemas críticos")

st.markdown(
    """
    A priorização combina a taxa de não resolução com o volume de avaliações.
    Uma taxa elevada baseada em poucas avaliações deve ser interpretada com cautela.
    """
)

if problemas_filtrados.empty:
    st.warning("Nenhum problema atende ao volume mínimo selecionado.")
else:
    top_problemas = problemas_filtrados.head(15).sort_values(
        "nao_resolvidas_pct",
        ascending=True,
    )

    fig_problemas = px.bar(
        top_problemas,
        x="nao_resolvidas_pct",
        y="Problema",
        orientation="h",
        color="nao_resolvidas_pct",
        color_continuous_scale=["#B7D7F0", AZUL, "#9B1C1C"],
        hover_data={"total_avaliado": True, "nao_resolvidas_pct": ":.2f"},
        title="Problemas com maior taxa de não resolução",
        labels={
            "Problema": "Tipo de problema",
            "nao_resolvidas_pct": "Não resolução (%)",
            "total_avaliado": "Avaliações válidas",
        },
    )

    fig_problemas.update_layout(
        yaxis_title="",
        xaxis_title="Não resolução (%)",
        coloraxis_showscale=False,
        height=600,
        margin=dict(l=10, r=10, t=60, b=20),
    )

    st.plotly_chart(fig_problemas, use_container_width=True)

    tabela_problemas = problemas_filtrados.rename(
        columns={
            "Problema": "Problema",
            "total_avaliado": "Avaliações válidas",
            "nao_resolvidas_pct": "Não resolução (%)",
        }
    ).copy()

    tabela_problemas["Não resolução (%)"] = (
        tabela_problemas["Não resolução (%)"].round(2)
    )

    st.dataframe(
        tabela_problemas,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# GRUPOS DE PROBLEMAS
# ============================================================

st.header("4. Desfecho por grupo de problema")

if not grupos.empty:
    colunas_grupos = grupos.columns.tolist()

    possiveis_nomes = [
        "Grupo Problema",
        "grupo_problema",
        "Grupo_Problema",
    ]

    coluna_grupo = next(
        (nome for nome in possiveis_nomes if nome in colunas_grupos),
        None,
    )

    possiveis_taxas = [
        "nao_resolvidas_pct",
        "nao_resolvidas_pct_sobre_avaliadas",
    ]

    coluna_taxa = next(
        (nome for nome in possiveis_taxas if nome in colunas_grupos),
        None,
    )

    if coluna_grupo and coluna_taxa:
        fig_grupos = px.bar(
            grupos.sort_values(coluna_taxa, ascending=False),
            x=coluna_grupo,
            y=coluna_taxa,
            title="Taxa de não resolução por grupo",
            labels={
                coluna_grupo: "Grupo",
                coluna_taxa: "Não resolução (%)",
            },
        )

        fig_grupos.update_layout(
            xaxis_title="Grupo de problema",
            yaxis_title="Não resolução (%)",
            xaxis_tickangle=-30,
        )

        st.plotly_chart(fig_grupos, use_container_width=True)
    else:
        st.caption(
            "Os nomes das colunas de grupo ou taxa são diferentes "
            "dos nomes esperados para este gráfico."
        )
        st.dataframe(grupos, use_container_width=True, hide_index=True)


# ============================================================
# RECOMENDAÇÕES
# ============================================================

st.header("5. Recomendações de atuação")

st.markdown(
    """
    As recomendações são regras de priorização derivadas dos problemas observados.
    Elas não demonstram, por si só, que uma ação aumentará a retenção.
    """
)

if not recomendacoes.empty:
    st.dataframe(
        recomendacoes,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Nenhuma recomendação disponível.")


# ============================================================
# PERFIS SINTÉTICOS
# ============================================================

st.header("6. Clientes sintéticos: valor, risco e ação")

st.warning(
    "DEMONSTRAÇÃO COM DADOS SINTÉTICOS: perfis, CLV, risco e ações abaixo "
    "não correspondem a passageiros reais e não devem orientar decisões operacionais."
)

if customer_profiles.empty:
    st.info(
        "A base sintética ainda não foi gerada. Execute `python "
        "src/synthetic/generate_passengers.py`, `python "
        "src/analysis/build_customer_intelligence.py` e reconstrua o DuckDB."
    )
else:
    segment_summary = (
        customer_profiles.groupby("segment", as_index=False)
        .agg(
            clientes=("customer_id", "count"),
            risco_medio=("churn_risk_score", "mean"),
            clv_medio=("clv_simulado_12m_brl", "mean"),
        )
    )
    fig_segments = px.bar(
        segment_summary,
        x="segment",
        y="clientes",
        color="risco_medio",
        hover_data={"clv_medio": ":,.2f", "risco_medio": ":.2%"},
        color_continuous_scale=[VERDE, "#F1C40F", VERMELHO],
        labels={"segment": "Segmento", "clientes": "Perfis sintéticos", "risco_medio": "Risco médio"},
        title="Perfis sintéticos por segmento",
    )
    fig_segments.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_segments, use_container_width=True)

    fig_matrix = px.scatter(
        customer_profiles,
        x="churn_risk_score",
        y="clv_simulado_12m_brl",
        color="segment",
        hover_name="customer_id",
        hover_data={"preferred_route_synthetic": True, "loyalty_tier_synthetic": True},
        labels={
            "churn_risk_score": "Risco estimado (dados sintéticos)",
            "clv_simulado_12m_brl": "CLV simulado em 12 meses (R$)",
            "segment": "Segmento",
        },
        title="Matriz de priorização: valor simulado × risco estimado",
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

    selected_customer = st.selectbox(
        "Selecione um passageiro sintético para a demonstração",
        customer_profiles["customer_id"].tolist(),
    )
    profile = customer_profiles.loc[
        customer_profiles["customer_id"] == selected_customer
    ].iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("CLV simulado (12 meses)", money(profile["clv_simulado_12m_brl"]))
    col2.metric("Risco estimado", percentage(profile["churn_risk_score"] * 100))
    col3.metric("Segmento", profile["segment"])
    st.caption(
        f"Tier sintético: {profile['loyalty_tier_synthetic']} · "
        f"Rota preferida sintética: {profile['preferred_route_synthetic']}"
    )
    st.write(f"**Fatores de risco demonstrativos:** {profile['risk_explanation']}")
    st.write(f"**Ação sugerida:** {profile['recommended_action']}")
    st.caption(profile["recommendation_rationale"])
    st.write(f"**Canal sugerido:** {profile['suggested_channel']}")
    st.write(f"**Mensagem rascunhada:** {profile['draft_message']}")
    st.caption(
        f"Custo demonstrativo da ação: {money(profile['estimated_action_cost_brl'])}. "
        "Aprovação humana e elegibilidade são obrigatórias antes de qualquer envio."
    )

    if SYNTHETIC_METRICS_PATH.exists():
        metrics = json.loads(SYNTHETIC_METRICS_PATH.read_text(encoding="utf-8"))
        st.caption(
            "Avaliação em dados sintéticos separados: "
            f"acurácia {metrics['accuracy']:.1%}, "
            f"precisão {metrics['precision']:.1%}, "
            f"recall {metrics['recall']:.1%}."
        )


# ============================================================
# ROI SIMULADO
# ============================================================

st.header("7. Simulação financeira de ações")

st.warning(
    "SIMULAÇÃO: taxas de retenção, unidades elegíveis, custos e margens "
    "são premissas hipotéticas. Não são resultados observados da Azul."
)

if not roi.empty:
    fig_roi = px.bar(
        roi,
        x="scenario_name",
        y="roi_pct",
        color="roi_pct",
        color_continuous_scale=["#C0392B", "#F1C40F", "#16865B"],
        title="ROI estimado por cenário",
        labels={
            "scenario_name": "Ação",
            "roi_pct": "ROI (%)",
        },
        hover_data={
            "investment_brl": ":,.2f",
            "gross_incremental_benefit_brl": ":,.2f",
            "net_benefit_brl": ":,.2f",
            "payback_months": ":.1f",
        },
    )

    fig_roi.update_layout(
        xaxis_title="Ação",
        yaxis_title="ROI (%)",
        coloraxis_showscale=False,
        xaxis_tickangle=-20,
    )

    st.plotly_chart(fig_roi, use_container_width=True)

    tabela_roi = roi[
        [
            "scenario_name",
            "investment_brl",
            "gross_incremental_benefit_brl",
            "net_benefit_brl",
            "roi_pct",
            "payback_months",
        ]
    ].copy()

    tabela_roi = tabela_roi.rename(
        columns={
            "scenario_name": "Cenário",
            "investment_brl": "Investimento (R$)",
            "gross_incremental_benefit_brl": "Benefício bruto (R$)",
            "net_benefit_brl": "Resultado líquido (R$)",
            "roi_pct": "ROI (%)",
            "payback_months": "Payback (meses)",
        }
    )

    st.dataframe(
        tabela_roi,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Premissas utilizadas na simulação"):
        st.dataframe(
            roi[
                [
                    "scenario_name",
                    "eligible_units_assumption",
                    "action_cost_per_unit_brl",
                    "fixed_campaign_cost_brl",
                    "assumed_incremental_retention_pct",
                    "contribution_margin_per_retained_customer_brl",
                    "benefit_horizon_months",
                    "retention_validated",
                ]
            ].rename(
                columns={
                    "scenario_name": "Cenário",
                    "eligible_units_assumption": "Unidades hipotéticas",
                    "action_cost_per_unit_brl": "Custo por unidade (R$)",
                    "fixed_campaign_cost_brl": "Custo fixo (R$)",
                    "assumed_incremental_retention_pct": "Retenção incremental assumida (%)",
                    "contribution_margin_per_retained_customer_brl": "Margem por cliente retido (R$)",
                    "benefit_horizon_months": "Horizonte (meses)",
                    "retention_validated": "Retenção validada?",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("Nenhum cenário financeiro foi encontrado.")


# ============================================================
# DIAGNÓSTICO DOCUMENTAL COM IA
# ============================================================

st.header("8. Diagnóstico com base documental")

st.markdown(
    """
    Faça perguntas sobre o programa de fidelidade documentado. A resposta usa
    apenas trechos recuperados da base local e identifica suas fontes.
    """
)

question = st.text_area(
    "Pergunta para o diagnóstico",
    placeholder="Ex.: Quais benefícios são mencionados e o que ainda precisa ser validado?",
)

if st.button("Gerar diagnóstico", type="primary"):
    try:
        result = answer_question(question)
    except ValueError as exc:
        st.warning(str(exc))
    except RuntimeError as exc:
        st.info(str(exc))
    else:
        st.markdown(result["answer"])

        if result["sources"]:
            st.caption("Evidências recuperadas para a resposta")
            st.dataframe(
                pd.DataFrame(result["sources"]).rename(
                    columns={
                        "id": "Referência",
                        "document": "Documento",
                        "section": "Seção",
                        "score": "Relevância",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# INFLUENCIADORES E PARCEIROS DE RELACIONAMENTO
# ============================================================

st.header("9. Influenciadores e parceiros de relacionamento")

st.markdown(
    """
    Curadoria transparente de criadores de conteúdo para ações regionais e campanhas de relacionamento.
    O ranking considera critérios objetivos: alinhamento temático com o programa de fidelidade,
    alcance médio e eficiência estimada de investimento (CPM).
    """
)

if influenciadores.empty:
    st.info("Nenhum parceiro ou influenciador catalogado no momento.")
else:
    col_rank1, col_rank2 = st.columns([2, 1])

    with col_rank1:
        fig_creators = px.bar(
            influenciadores,
            x="creator_name",
            y="fit_score",
            color="fit_score",
            color_continuous_scale=["#B7D7F0", AZUL, "#16865B"],
            title="Score de adequação por criador de conteúdo",
            labels={
                "creator_name": "Criador / Canal",
                "fit_score": "Score de adequação (0 a 1)",
            },
            hover_data={
                "focus_theme": True,
                "subscriber_tier": True,
                "avg_views_per_video": ":,",
                "suggested_partnership": True,
            },
        )
        fig_creators.update_layout(xaxis_tickangle=-15, coloraxis_showscale=False)
        st.plotly_chart(fig_creators, use_container_width=True)

    with col_rank2:
        st.markdown("**Critérios de ponderação transparente:**")
        st.write("- **50%**: Alinhamento temático com fidelidade e aviação.")
        st.write("- **30%**: Volume médio de visualizações por publicação.")
        st.write("- **20%**: Eficiência estimada de custo por milhar (CPM).")
        st.caption(
            "Popularidade não garante conversão direta. As propostas de ação "
            "são hipóteses de engajamento a serem validadas."
        )

    st.subheader("Painel de parcerias sugeridas")
    tabela_influenciadores = influenciadores[
        [
            "ranking",
            "creator_name",
            "focus_theme",
            "primary_platform",
            "loyalty_focus",
            "fit_score",
            "suggested_partnership",
        ]
    ].rename(
        columns={
            "ranking": "Posição",
            "creator_name": "Criador / Canal",
            "focus_theme": "Foco Temático",
            "primary_platform": "Plataforma Principal",
            "loyalty_focus": "Enfoque em Fidelidade",
            "fit_score": "Score de Adequação",
            "suggested_partnership": "Ação de Parceria Sugerida",
        }
    )
    st.dataframe(tabela_influenciadores, use_container_width=True, hide_index=True)


# ============================================================
# RODAPÉ
# ============================================================

st.divider()

st.caption(
    "Radar de Alto Valor — protótipo de hackathon. "
    "Fonte pública: ANAC / Consumidor.gov.br, dados de 2025. "
    "Indicadores de reclamações são observados; projeções financeiras são simuladas. "
    "Não há inferência individual de churn ou CLV sem dados transacionais apropriados."
)
