"""Painel Executivo e Cockpit de CRM Inteligente — Company Smart Loyalty (Azul Fidelidade).

Integra:
- F1/F2: Indicadores ANAC, Reclamações, Score de Alto Valor e CLV Preditivo
- F3/F4: Modelo de Evasão (Churn), Explicabilidade SHAP Local e 5 Personas Acionáveis
- F5: Agente Cognitivo de CRM com RAG nas regras da Azul e Guardrails (Zero Alucinações)
- F6: Cockpit Interativo de CRM, Matriz Valor x Risco, Ficha 360° e Human-in-the-Loop
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configuração de caminhos do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agent.action_agent import ActionAgent, validate_benefits_guardrails
from src.analysis.build_analytics_db import create_database
from src.analysis.process_anac_year import process_year
from src.ingestion.update_anac_dataset import update_dataset
from src.knowledge.answer_question import answer_question

# ============================================================
# CONFIGURAÇÃO GERAL DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Radar de Alto Valor | Azul Fidelidade",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta visual corporativa Azul Fidelidade
AZUL_MARINHO = "#00205B"
AZUL_ROYAL = "#00529B"
AZUL_CEU = "#00A3E0"
AZUL_CLARO = "#EAF3FB"
VERDE_RETENCAO = "#16865B"
AMARELO_ALERTA = "#F1C40F"
VERMELHO_CHURN = "#C0392B"
CINZA_TEXTO = "#667085"

DB_PATH = PROJECT_ROOT / "data" / "analytics" / "company_analytics.duckdb"
SYNTHETIC_METRICS_PATH = PROJECT_ROOT / "data" / "gold" / "metricas_modelo_sintetico.json"

# Inicialização de variáveis de sessão (Human-in-the-Loop)
if "approved_actions" not in st.session_state:
    st.session_state["approved_actions"] = []
if "generated_recommendations" not in st.session_state:
    st.session_state["generated_recommendations"] = {}
if "selected_customer_id" not in st.session_state:
    st.session_state["selected_customer_id"] = "SYN-00763"


# ============================================================
# FUNÇÕES UTILITÁRIAS DE ACESSO E FORMATAÇÃO
# ============================================================

@st.cache_resource
def get_action_agent() -> ActionAgent:
    """Instancia o Agente Cognitivo de CRM com a base RAG indexada."""
    return ActionAgent()


@st.cache_data(ttl=300)
def read_query(query: str) -> pd.DataFrame:
    """Executa uma consulta somente leitura no DuckDB."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Banco não encontrado: {DB_PATH}. Execute primeiro build_analytics_db.py."
        )
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return connection.execute(query).df()
    finally:
        connection.close()


def money(value) -> str:
    """Formata valores em reais (R$)."""
    if pd.isna(value) or value is None:
        return "R$ —"
    return f"R$ {float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def percentage(value) -> str:
    """Formata valores em percentual (%)."""
    if pd.isna(value) or value is None:
        return "—"
    return f"{float(value):.1f}%".replace(".", ",")


# ============================================================
# CARREGAMENTO DOS DADOS ANALÍTICOS
# ============================================================

if not DB_PATH.exists():
    st.error(
        "Banco analítico não encontrado. Execute "
        "`python src/analysis/build_analytics_db.py` antes de abrir o dashboard."
    )
    st.stop()

try:
    indicadores = read_query("SELECT * FROM indicadores_2025 LIMIT 1")
    mensal = read_query("SELECT mes, reclamacoes FROM reclamacoes_mensais_2025")
    problemas = read_query(
        """
        SELECT Problema, total_avaliado, nao_resolvidas_pct
        FROM desfecho_por_problema_2025
        WHERE total_avaliado >= 30
        ORDER BY nao_resolvidas_pct DESC
        """
    )
    grupos = read_query("SELECT * FROM desfecho_por_grupo_2025")
    roi = read_query("SELECT * FROM roi_scenarios ORDER BY roi_pct DESC")
    recomendacoes = read_query("SELECT * FROM recomendacoes_2025")
    customer_profiles = read_query("SELECT * FROM perfis_clientes_sinteticos")
    influenciadores = read_query("SELECT * FROM influenciadores_viagem ORDER BY ranking ASC")
except Exception as exc:
    st.error(f"Erro ao carregar tabelas do DuckDB: {exc}")
    st.stop()


# Classificação de Quadrantes Valor x Risco
if not customer_profiles.empty:
    clv_median = float(customer_profiles["clv_simulado_12m_brl"].median())
    risk_cutoff = 0.50

    def assign_quadrant(row):
        val_high = row["clv_simulado_12m_brl"] >= clv_median
        risk_high = row["churn_risk_score"] >= risk_cutoff
        if val_high and risk_high:
            return "Alto Valor × Alto Risco"
        elif val_high and not risk_high:
            return "Alto Valor × Baixo Risco"
        elif not val_high and risk_high:
            return "Baixo Valor × Alto Risco"
        else:
            return "Baixo Valor × Baixo Risco"

    customer_profiles["quadrante"] = customer_profiles.apply(assign_quadrant, axis=1)


# ============================================================
# BARRA LATERAL (FILTROS GLOBAIS E DE CRM)
# ============================================================

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Azul_Linhas_A%C3%A9reas_Brasileiras_logo.svg/320px-Azul_Linhas_A%C3%A9reas_Brasileiras_logo.svg.png",
    use_container_width=True,
)
st.sidebar.title("Radar de Alto Valor")
st.sidebar.caption("Inteligência Preditiva & CRM Cognitivo — Azul Fidelidade")

st.sidebar.divider()
st.sidebar.subheader("🎯 Filtros da Carteira de Clientes")

# Filtros para o Cockpit de CRM
available_personas = sorted(customer_profiles["persona"].unique().tolist()) if not customer_profiles.empty else []
selected_personas = st.sidebar.multiselect(
    "Personas da Aviação",
    options=available_personas,
    default=available_personas,
    help="Segmentação baseada nos clusters K-Means da Fase 1.",
)

tier_order = ["Diamante", "Safira", "Topázio", "Básico"]
available_tiers = [t for t in tier_order if t in customer_profiles["loyalty_tier_synthetic"].unique().tolist()] if not customer_profiles.empty else []
selected_tiers = st.sidebar.multiselect(
    "Tier Azul Fidelidade",
    options=available_tiers,
    default=available_tiers,
    help="Categorias oficiais do programa.",
)

quadrant_options = [
    "Todos os Quadrantes",
    "Alto Valor × Alto Risco",
    "Alto Valor × Baixo Risco",
    "Baixo Valor × Alto Risco",
    "Baixo Valor × Baixo Risco",
]
selected_quadrant = st.sidebar.selectbox(
    "Quadrante Valor × Risco",
    options=quadrant_options,
    index=0,
)

prioritize_critical = st.sidebar.checkbox(
    "🚨 Foco Imediato: Alto Valor × Alto Risco",
    value=False,
    help="Filtra automaticamente clientes com alto CLV e risco iminente de churn.",
)

# Aplicação dos filtros no dataframe de trabalho
filtered_profiles = customer_profiles.copy()
if selected_personas:
    filtered_profiles = filtered_profiles[filtered_profiles["persona"].isin(selected_personas)]
if selected_tiers:
    filtered_profiles = filtered_profiles[filtered_profiles["loyalty_tier_synthetic"].isin(selected_tiers)]

if prioritize_critical:
    filtered_profiles = filtered_profiles[filtered_profiles["quadrante"] == "Alto Valor × Alto Risco"]
elif selected_quadrant != "Todos os Quadrantes":
    filtered_profiles = filtered_profiles[filtered_profiles["quadrante"] == selected_quadrant]

# Seção técnica da ANAC na barra lateral
st.sidebar.divider()
with st.sidebar.expander("⚙️ Pipeline de Dados ANAC"):
    st.caption("Atualização da fonte pública de reclamações")
    selected_data_year = st.number_input(
        "Ano dos dados ANAC",
        min_value=2014,
        max_value=datetime.now().year,
        value=2025,
        step=1,
    )
    if st.button("Validar e baixar nova base", key="btn_download_anac"):
        try:
            with st.spinner("Baixando fonte ANAC..."):
                up_meta = update_dataset(int(selected_data_year))
                st.success(f"Base {up_meta['year']} validada!")
        except Exception as err:
            st.error(f"Erro: {err}")


# ============================================================
# CABEÇALHO PRINCIPAL
# ============================================================

st.title("✈️ Radar de Alto Valor — Azul Fidelidade")
st.markdown(
    "**Plataforma Integrada de Inteligência de Clientes:** da detecção analítica de riscos operacionais (ANAC) "
    "à prescrição de retenção personalizada ancorada estritamente nas regras públicas do **Azul Fidelidade**."
)


# ============================================================
# NAVEGAÇÃO PRINCIPAL EM ABAS
# ============================================================

tab_crm, tab_anac, tab_roi, tab_rag, tab_creators = st.tabs([
    "🎯 Cockpit de CRM & Agente Cognitivo",
    "📊 Indicadores ANAC & Operacionais",
    "💰 Simulação de ROI & Negócios",
    "📚 RAG & Diagnóstico Documental",
    "⭐ Parcerias & Influenciadores",
])


# ======================================================================================
# ABA 1: COCKPIT DE CRM & AGENTE COGNITIVO (FASE 3 — ESTRELA DO PITCH)
# ======================================================================================
with tab_crm:
    # Header mandatório para compatibilidade com testes existentes
    st.header("6. Clientes sintéticos: valor, risco e ação")

    st.markdown(
        """
        Visão completa da esteira de CRM Cognitivo: **Matriz de Priorização**, **Ficha 360° com Explicabilidade SHAP**, 
        **Geração da Próxima Melhor Ação com RAG** e **Fila de Disparo Human-in-the-Loop**.
        """
    )

    if filtered_profiles.empty:
        st.warning("Nenhum cliente encontrado com os filtros selecionados. Ajuste os filtros na barra lateral.")
    else:
        # Métricas Globais da Carteira Filtrada
        total_filtered = len(filtered_profiles)
        high_risk_count = len(filtered_profiles[filtered_profiles["churn_risk_score"] >= 0.50])
        total_clv_portfolio = filtered_profiles["clv_simulado_12m_brl"].sum()
        revenue_at_risk = filtered_profiles[filtered_profiles["churn_risk_score"] >= 0.50]["clv_simulado_12m_brl"].sum()

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Carteira em Análise", f"{total_filtered:,} clientes".replace(",", "."))
        mcol2.metric("Passageiros em Alto Risco", f"{high_risk_count:,} ({high_risk_count/max(1, total_filtered):.1%})".replace(".", ","), delta=f"-{high_risk_count} sob ameaça", delta_color="inverse")
        mcol3.metric("CLV Total da Carteira", money(total_clv_portfolio))
        mcol4.metric("Receita em Risco Imediato", money(revenue_at_risk), delta="Receita a blindar", delta_color="inverse")

        st.divider()

        # ----------------------------------------------------
        # 1. MATRIZ VALOR × RISCO (VISÃO DA CARTEIRA)
        # ----------------------------------------------------
        st.subheader("1. Matriz de Priorização Estratégica: Valor Projetado (CLV) × Risco de Churn")

        color_map = {
            "Alto Valor em Risco": VERMELHO_CHURN,
            "Executivo Frequente": AZUL_ROYAL,
            "Lazer Premium": "#9B59B6",
            "Conectividade Regional": AZUL_CEU,
            "Ocasional em Ascensão": CINZA_TEXTO,
        }

        fig_matrix = px.scatter(
            filtered_profiles,
            x="churn_risk_score",
            y="clv_simulado_12m_brl",
            color="persona",
            color_discrete_map=color_map,
            hover_name="customer_id",
            hover_data={
                "loyalty_tier_synthetic": True,
                "persona": True,
                "clv_simulado_12m_brl": ":,.2f",
                "churn_risk_score": ":.1%",
                "preferred_route_synthetic": True,
                "quadrante": True,
            },
            labels={
                "churn_risk_score": "Probabilidade de Churn (Modelo Tabular)",
                "clv_simulado_12m_brl": "CLV Projetado em 12 Meses (R$)",
                "persona": "Persona",
            },
            title="Matriz de Clientes: Identificação de Casos Críticos de Retenção",
            height=480,
        )

        # Linhas divisórias dos 4 quadrantes
        fig_matrix.add_vline(x=0.50, line_dash="dash", line_color="#BDC3C7", line_width=1.5)
        fig_matrix.add_hline(y=clv_median, line_dash="dash", line_color="#BDC3C7", line_width=1.5)

        # Anotações dos quadrantes
        fig_matrix.add_annotation(
            x=0.85, y=customer_profiles["clv_simulado_12m_brl"].max() * 0.92,
            text="🚨 <b>ALTO VALOR × ALTO RISCO</b><br>Foco Prioritário de Retenção",
            showarrow=False, bgcolor="rgba(192, 57, 43, 0.12)", bordercolor=VERMELHO_CHURN, borderwidth=1,
            font=dict(color=VERMELHO_CHURN, size=11),
        )
        fig_matrix.add_annotation(
            x=0.15, y=customer_profiles["clv_simulado_12m_brl"].max() * 0.92,
            text="💎 <b>ALTO VALOR × BAIXO RISCO</b><br>Fidelizados & Blindagem VIP",
            showarrow=False, bgcolor="rgba(0, 82, 155, 0.10)", bordercolor=AZUL_ROYAL, borderwidth=1,
            font=dict(color=AZUL_ROYAL, size=11),
        )

        fig_matrix.update_layout(
            xaxis=dict(tickformat=".0%"),
            yaxis=dict(tickprefix="R$ "),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
        )

        st.plotly_chart(fig_matrix, use_container_width=True)

        st.divider()

        # ----------------------------------------------------
        # 2. FICHA 360° DO CLIENTE & EXPLICABILIDADE SHAP
        # ----------------------------------------------------
        st.subheader("2. Ficha 360° do Cliente & Explicabilidade Individual SHAP")

        # Seleção de cliente
        customer_options = filtered_profiles["customer_id"].tolist()
        
        # Garantir cliente default se estiver nos filtrados
        default_index = 0
        if st.session_state["selected_customer_id"] in customer_options:
            default_index = customer_options.index(st.session_state["selected_customer_id"])

        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            selected_id = st.selectbox(
                "Selecione um passageiro para investigação profunda e acionamento:",
                options=customer_options,
                index=default_index,
                format_func=lambda cid: (
                    f"{cid} | Tier: {filtered_profiles.loc[filtered_profiles['customer_id'] == cid, 'loyalty_tier_synthetic'].values[0]} "
                    f"| Persona: {filtered_profiles.loc[filtered_profiles['customer_id'] == cid, 'persona'].values[0]} "
                    f"| CLV: {money(filtered_profiles.loc[filtered_profiles['customer_id'] == cid, 'clv_simulado_12m_brl'].values[0])} "
                    f"| Churn: {filtered_profiles.loc[filtered_profiles['customer_id'] == cid, 'churn_risk_score'].values[0]:.1%}"
                ),
            )
            st.session_state["selected_customer_id"] = selected_id

        with col_sel2:
            st.info(
                f"💡 **Dica de Demonstração:** Selecione `SYN-00763` (Safira em Risco) ou "
                f"`SYN-00003` (Executivo Diamante) para simular casos reais do pitch."
            )

        client_row = customer_profiles[customer_profiles["customer_id"] == selected_id].iloc[0]

        # Painel de Métricas 360°
        tier_str = client_row["loyalty_tier_synthetic"]
        tier_badges = {
            "Diamante": "💎 **DIAMANTE** (Tier Máximo)",
            "Safira": "🔷 **SAFIRA** (Tier Intermediário)",
            "Topázio": "🔶 **TOPÁZIO** (Tier Frequente)",
            "Básico": "⚪ **BÁSICO** (Nível de Entrada)",
        }

        with st.container(border=True):
            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            pcol1.markdown(f"**Categoria Fidelidade:**\n### {tier_badges.get(tier_str, tier_str)}")
            pcol2.metric("CLV Projetado (12M)", money(client_row["clv_simulado_12m_brl"]))
            pcol3.metric(
                "Risco de Churn",
                f"{client_row['churn_risk_score']:.1%}",
                delta="Crítico" if client_row['churn_risk_score'] >= 0.5 else "Estável",
                delta_color="inverse" if client_row['churn_risk_score'] >= 0.5 else "normal",
            )
            pcol4.metric("Saldo de Pontos", f"{int(client_row['points_balance']):,}".replace(",", "."))

            st.divider()

            qcol1, qcol2, qcol3, qcol4 = st.columns(4)
            qcol1.write(f"**Persona:** `{client_row['persona']}`")
            qcol1.write(f"**Quadrante:** `{client_row['quadrante']}`")

            qcol2.write(f"**Trechos Voados (12m):** {int(client_row['flights_last_12m'])} voos")
            qcol2.write(f"**Trechos Anteriores:** {int(client_row['flights_previous_12m'])} voos")

            qcol3.write(f"**Recência sem Voar:** {int(client_row['months_since_last_flight'])} meses")
            qcol3.write(f"**Reclamações Pendentes:** {int(client_row.get('unresolved_issues_last_12m', 0))}")

            qcol4.write(f"**Rota Principal:** `{client_row['preferred_route_synthetic']}`")
            qcol4.write(f"**Margem Média da Rota:** {client_row.get('route_margin_avg', 20.0):.1f}%")

        # Gráfico SHAP Local
        st.markdown("#### 🔬 Explicabilidade SHAP Local: O que está influenciando a probabilidade de evasão?")

        shap_json_str = client_row.get("top_shap_values_json", "[]")
        shap_factors = []
        try:
            shap_factors = json.loads(shap_json_str)
        except Exception:
            shap_factors = []

        if shap_factors:
            shap_df = pd.DataFrame(shap_factors)
            shap_df["impacto"] = shap_df["shap_value"].apply(
                lambda val: "Aumenta Risco de Evasão (+)" if val > 0 else "Fator Retentor (-)"
            )
            shap_df = shap_df.sort_values("shap_value", ascending=True)

            fig_shap = px.bar(
                shap_df,
                x="shap_value",
                y="label",
                orientation="h",
                color="impacto",
                color_discrete_map={
                    "Aumenta Risco de Evasão (+)": VERMELHO_CHURN,
                    "Fator Retentor (-)": VERDE_RETENCAO,
                },
                labels={"shap_value": "Contribuição para o Log-Odds de Evasão (SHAP)", "label": "Driver Comportamental"},
                title=f"Drivers de Risco Individuais — Passageiro {selected_id}",
                height=260,
            )
            fig_shap.update_layout(margin=dict(l=20, r=20, t=40, b=20), legend_title_text="")
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info(f"**Fatores Identificados:** {client_row.get('risk_explanation', 'Sem alertas críticos')}")

        st.divider()

        # ----------------------------------------------------
        # 3. INTEGRAÇÃO COM AGENTE COGNITIVO (NEXT BEST ACTION)
        # ----------------------------------------------------
        st.subheader("3. Agente Cognitivo de CRM: Próxima Melhor Ação com RAG & Guardrails")

        agent = get_action_agent()

        action_container = st.container(border=True)

        with action_container:
            btn_col1, btn_col2 = st.columns([2, 1])
            with btn_col1:
                st.markdown(
                    "Clique no botão para acionar o motor de recomendação em tempo real. "
                    "O agente consultará as regras oficiais do regulamento via RAG e aplicará os guardrails de zero alucinações."
                )
            with btn_col2:
                generate_clicked = st.button(
                    "🤖 Gerar Próxima Melhor Ação (RAG)",
                    type="primary",
                    use_container_width=True,
                )

            # Se o botão foi clicado ou se já temos recomendação em cache para este cliente
            cache_key = f"rec_{selected_id}"
            if generate_clicked or cache_key in st.session_state["generated_recommendations"]:
                if generate_clicked:
                    with st.spinner("Consultando regulamento oficial do Azul Fidelidade e aplicando guardrails..."):
                        recommendation = agent.generate_action(client_row.to_dict())
                        st.session_state["generated_recommendations"][cache_key] = recommendation.model_dump()

                rec_data = st.session_state["generated_recommendations"][cache_key]

                st.success(f"🎯 **Ação Prescrita:** {rec_data['acao_proposta']}")

                rcol1, rcol2, rcol3 = st.columns(3)
                rcol1.markdown(f"**Canal Recomendado:**\n`{rec_data['canal_recomendado']}`")
                rcol2.markdown(f"**Custo Estimado da Concessão:**\n`{money(rec_data['custo_estimado_reais'])}`")
                rcol3.markdown(f"**Status de Governança:**\n`{rec_data['status_aprovacao']}`")

                # Citação de Regra com destaque
                st.info(f"📜 **Fundamentação Regulatória Auditada:**\n{rec_data['regra_citada']}")

                # Mensagem personalizada pronta com edição do analista
                st.markdown("**Mensagem Empática Personalizada (Pronta para envio com opção de edição):**")
                edited_message = st.text_area(
                    "Texto da Mensagem:",
                    value=rec_data["mensagem_personalizada"],
                    height=100,
                    key=f"msg_edit_{selected_id}",
                )

                # Botão Human-in-the-Loop de Aprovação
                if st.button("✅ Aprovar Ação e Adicionar à Fila de Disparo (Human-in-the-Loop)", type="secondary", use_container_width=True):
                    # Verificar se já foi aprovado para evitar duplicidade
                    approved_ids = [item["cliente_id"] for item in st.session_state["approved_actions"]]
                    if selected_id not in approved_ids:
                        approved_item = {
                            "cliente_id": selected_id,
                            "tier": tier_str,
                            "persona": client_row["persona"],
                            "clv_protegido_brl": float(client_row["clv_simulado_12m_brl"]),
                            "acao_proposta": rec_data["acao_proposta"],
                            "canal": rec_data["canal_recomendado"],
                            "custo_reais": float(rec_data["custo_estimado_reais"]),
                            "regra_citada": rec_data["regra_citada"],
                            "mensagem_final": edited_message,
                            "data_aprovacao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "Aprovado para Disparo",
                        }
                        st.session_state["approved_actions"].append(approved_item)
                        st.toast(f"Ação para o cliente {selected_id} aprovada com sucesso!", icon="✅")
                        st.rerun()
                    else:
                        st.warning(f"O passageiro {selected_id} já consta na fila de ações aprovadas.")

        st.divider()

        # ----------------------------------------------------
        # 4. FILA DE AÇÕES APROVADAS & EXPORTAÇÃO DE LOTE
        # ----------------------------------------------------
        st.subheader("4. Fila de Ações Aprovadas & Exportação de Lote de CRM (Human-in-the-Loop)")

        approved_list = st.session_state["approved_actions"]

        if not approved_list:
            st.info(
                "A fila de ações aprovadas está vazia nesta sessão. "
                "Selecione um passageiro no quadrante acima e clique em **Aprovar Ação** para compor a campanha de CRM."
            )
        else:
            app_df = pd.DataFrame(approved_list)

            tot_approved = len(app_df)
            tot_cost = app_df["custo_reais"].sum()
            tot_protected_clv = app_df["clv_protegido_brl"].sum()
            roi_campaign = ((tot_protected_clv - tot_cost) / max(1.0, tot_cost)) * 100.0

            acol1, acol2, acol3, acol4 = st.columns(4)
            acol1.metric("Ações Aprovadas", f"{tot_approved} clientes")
            acol2.metric("Custo Total da Campanha", money(tot_cost))
            acol3.metric("Receita Protegida (CLV)", money(tot_protected_clv))
            acol4.metric("ROI Estimado da Ação", f"{roi_campaign:,.1f}%".replace(",", "X").replace(".", ",").replace("X", "."))

            st.dataframe(
                app_df[[
                    "cliente_id", "tier", "persona", "acao_proposta", "canal",
                    "custo_reais", "regra_citada", "data_aprovacao", "status"
                ]].rename(columns={
                    "cliente_id": "Cliente ID",
                    "tier": "Tier",
                    "persona": "Persona",
                    "acao_proposta": "Ação Proposta",
                    "canal": "Canal",
                    "custo_reais": "Custo (R$)",
                    "regra_citada": "Regra Regulamentar",
                    "data_aprovacao": "Horário de Aprovação",
                    "status": "Status",
                }),
                use_container_width=True,
                hide_index=True,
            )

            # Exportação CSV para Salesforce / Braze
            csv_export = app_df.to_csv(index=False, encoding="utf-8-sig")
            exp_col1, exp_col2 = st.columns([3, 1])

            with exp_col1:
                st.download_button(
                    label="📥 Exportar Lote para Disparo de CRM (CSV — Braze / Salesforce / WhatsApp)",
                    data=csv_export,
                    file_name=f"lote_retencao_crm_azul_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with exp_col2:
                if st.button("🗑️ Limpar Fila da Sessão", use_container_width=True):
                    st.session_state["approved_actions"] = []
                    st.rerun()


# ======================================================================================
# ABA 2: INDICADORES ANAC & OPERACIONAIS (PRESERVANDO SEÇÕES 1 A 5 ORIGINAIS)
# ======================================================================================
with tab_anac:
    st.header("1. Visão executiva")
    dados = indicadores.iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Reclamações", f"{int(dados['total_reclamacoes']):,}".replace(",", "."))
    col2.metric("Respondidas", percentage(dados["reclamacoes_respondidas_pct"]))
    col3.metric(
        "Não resolução",
        percentage(dados["nao_resolvidas_pct_sobre_avaliadas"]),
        help="Percentual entre reclamações avaliadas como resolvidas ou não resolvidas.",
    )
    col4.metric("Resposta média", f"{dados['tempo_medio_resposta_dias']:.1f} dias")
    col5.metric("Nota média", f"{dados['nota_media_consumidor']:.2f}/5")

    st.caption(
        f"Avaliações válidas: {int(dados['reclamacoes_avaliadas']):,}. "
        f"Resolvidas: {int(dados['reclamacoes_resolvidas']):,}. "
        f"Não resolvidas: {int(dados['reclamacoes_nao_resolvidas']):,}."
        .replace(",", ".")
    )

    st.divider()

    st.header("2. Evolução mensal das reclamações")
    if not mensal.empty:
        ordem_meses = {
            "Janeiro": 1, "Fevereiro": 2, "Março": 3, "Abril": 4, "Maio": 5, "Junho": 6,
            "Julho": 7, "Agosto": 8, "Setembro": 9, "Outubro": 10, "Novembro": 11, "Dezembro": 12,
        }
        mensal_ordenado = mensal.copy()
        mensal_ordenado["ordem_mes"] = mensal_ordenado["mes"].map(ordem_meses).fillna(99)
        mensal_ordenado = mensal_ordenado.sort_values("ordem_mes")

        fig_mensal = px.line(
            mensal_ordenado,
            x="mes",
            y="reclamacoes",
            markers=True,
            title="Volume mensal de reclamações — 2025",
            labels={"mes": "Mês", "reclamacoes": "Reclamações"},
        )
        fig_mensal.update_traces(line=dict(color=AZUL_ROYAL, width=3), marker=dict(size=8))
        st.plotly_chart(fig_mensal, use_container_width=True)

    st.divider()

    st.header("3. Principais motivos de reclamação")
    if not problemas.empty:
        fig_prob = px.bar(
            problemas.head(10),
            x="nao_resolvidas_pct",
            y="Problema",
            orientation="h",
            color="nao_resolvidas_pct",
            color_continuous_scale=[VERDE_RETENCAO, AMARELO_ALERTA, VERMELHO_CHURN],
            title="Top 10 problemas por taxa de não resolução (mínimo 30 avaliações)",
            labels={"nao_resolvidas_pct": "Não resolvidas (%)", "Problema": "Motivo"},
        )
        fig_prob.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig_prob, use_container_width=True)

    st.divider()

    st.header("4. Detalhamento por grupo de problemas")
    if not grupos.empty:
        st.dataframe(
            grupos.rename(columns={
                "Grupo Problema": "Grupo",
                "total_reclamacoes": "Reclamações",
                "respondidas_pct": "Respondidas (%)",
                "nao_resolvidas_pct": "Não Resolvidas (%)",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    st.header("5. Recomendações e plano de ação")
    if not recomendacoes.empty:
        st.dataframe(
            recomendacoes[[
                "Problema", "total_avaliado", "nao_resolvidas_pct", "acao_sugerida", "justificativa"
            ]].rename(columns={
                "Problema": "Problema Identificado",
                "total_avaliado": "Total Avaliado",
                "nao_resolvidas_pct": "Não Resolução (%)",
                "acao_sugerida": "Ação Operacional",
                "justificativa": "Justificativa",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ======================================================================================
# ABA 3: SIMULAÇÃO FINANCEIRA DE ROI (PRESERVANDO SEÇÃO 7 ORIGINAL)
# ======================================================================================
with tab_roi:
    st.header("7. Simulação financeira de ações")
    st.warning(
        "SIMULAÇÃO: taxas de retenção, unidades elegíveis, custos e margens "
        "são premissas hipotéticas baseadas em cenários de negócio."
    )

    if not roi.empty:
        fig_roi = px.bar(
            roi,
            x="scenario_name",
            y="roi_pct",
            color="roi_pct",
            color_continuous_scale=[VERMELHO_CHURN, AMARELO_ALERTA, VERDE_RETENCAO],
            title="ROI estimado por cenário de retenção",
            labels={"scenario_name": "Ação", "roi_pct": "ROI (%)"},
            hover_data={
                "investment_brl": ":,.2f",
                "gross_incremental_benefit_brl": ":,.2f",
                "net_benefit_brl": ":,.2f",
                "payback_months": ":.1f",
            },
        )
        fig_roi.update_layout(xaxis_title="Cenário", yaxis_title="ROI (%)", coloraxis_showscale=False)
        st.plotly_chart(fig_roi, use_container_width=True)

        st.dataframe(
            roi[[
                "scenario_name", "investment_brl", "gross_incremental_benefit_brl",
                "net_benefit_brl", "roi_pct", "payback_months"
            ]].rename(columns={
                "scenario_name": "Cenário de CRM",
                "investment_brl": "Investimento (R$)",
                "gross_incremental_benefit_brl": "Benefício Bruto (R$)",
                "net_benefit_brl": "Resultado Líquido (R$)",
                "roi_pct": "ROI (%)",
                "payback_months": "Payback (meses)",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ======================================================================================
# ABA 4: RAG & DIAGNÓSTICO DOCUMENTAL (PRESERVANDO SEÇÃO 8 ORIGINAL)
# ======================================================================================
with tab_rag:
    st.header("8. Diagnóstico com base documental")
    st.markdown(
        "Faça perguntas sobre o programa de fidelidade documentado. A resposta usa "
        "apenas trechos recuperados da base local e identifica suas fontes."
    )

    question = st.text_area(
        "Pergunta para o diagnóstico regulamentar:",
        placeholder="Ex.: Quais os requisitos de qualificação do Tier Safira e Diamante e franquias de bagagem?",
        key="rag_chat_question",
    )

    if st.button("Gerar diagnóstico RAG", type="primary"):
        try:
            result = answer_question(question)
        except Exception as exc:
            st.error(f"Erro na consulta: {exc}")
        else:
            st.markdown(result["answer"])
            if result.get("sources"):
                st.caption("Evidências regulamentares recuperadas:")
                st.dataframe(
                    pd.DataFrame(result["sources"]).rename(columns={
                        "id": "Referência",
                        "document": "Documento",
                        "section": "Seção",
                        "score": "Relevância",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )


# ======================================================================================
# ABA 5: PARCERIAS & INFLUENCIADORES (PRESERVANDO SEÇÃO 9 ORIGINAL)
# ======================================================================================
with tab_creators:
    st.header("9. Influenciadores e parceiros de relacionamento")
    st.markdown(
        "Curadoria transparente de criadores de conteúdo para ações regionais e campanhas de relacionamento. "
        "O ranking considera alinhamento temático com o programa de fidelidade, alcance médio e eficiência estimada (CPM)."
    )

    if not influenciadores.empty:
        fig_creators = px.bar(
            influenciadores,
            x="creator_name",
            y="fit_score",
            color="fit_score",
            color_continuous_scale=[AZUL_CLARO, AZUL_ROYAL, VERDE_RETENCAO],
            title="Score de adequação por criador de conteúdo",
            labels={"creator_name": "Criador / Canal", "fit_score": "Score de Adequação"},
        )
        st.plotly_chart(fig_creators, use_container_width=True)

        st.dataframe(
            influenciadores[[
                "ranking", "creator_name", "focus_theme", "primary_platform",
                "loyalty_focus", "fit_score", "suggested_partnership"
            ]].rename(columns={
                "ranking": "Posição",
                "creator_name": "Criador / Canal",
                "focus_theme": "Foco Temático",
                "primary_platform": "Plataforma Principal",
                "loyalty_focus": "Enfoque em Fidelidade",
                "fit_score": "Score de Adequação",
                "suggested_partnership": "Ação de Parceria Sugerida",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# RODAPÉ INSTITUCIONAL
# ============================================================

st.divider()
st.caption(
    "Radar de Alto Valor — Solução desenvolvida para o Hackathon Avanade / Azul Linhas Aéreas. "
    "Indicadores de reclamações: ANAC / Consumidor.gov.br (2025). "
    "Políticas de fidelidade: Regulamento Oficial do Programa Azul Fidelidade. "
    "Inteligência de CRM com Machine Learning, SHAP e RAG Guardrails (Zero Alucinações)."
)
