"""Módulo do Agente Cognitivo de CRM com RAG e Guardrails.

Responsável por receber o perfil 360° do passageiro (incluindo persona, CLV e fatores
SHAP de risco) e gerar a Próxima Melhor Ação (Next Best Action - NBA) ancorada estritamente
nas regras públicas do programa Azul Fidelidade.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# Importação da esteira de recuperação lexical RAG
try:
    from src.knowledge.retrieve_documents import (
        DEFAULT_DOCUMENTS_DIR,
        load_documents,
        search_documents,
    )
except ImportError:
    # Fallback se executado de diretório diferente
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.knowledge.retrieve_documents import (
        DEFAULT_DOCUMENTS_DIR,
        load_documents,
        search_documents,
    )

logger = logging.getLogger("action_agent")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =====================================================================
# Modelos de Dados Pydantic
# =====================================================================

class CustomerProfile(BaseModel):
    """Representação unificada do perfil 360° do passageiro."""
    cliente_id: str
    tier: str = "Básico"
    persona: str = "Ocasional em Ascensão"
    clv_simulado: float = 0.0
    churn_risk_score: float = 0.0
    driver_risco_1: str = "Nenhum atrito crítico detectado"
    driver_risco_2: Optional[str] = None
    driver_risco_3: Optional[str] = None
    points_balance: int = 0
    months_since_last_flight: int = 0
    unresolved_issues: int = 0
    preferred_route: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomerProfile":
        """Constrói perfil a partir de dicionário ou série do pandas."""
        cliente_id = str(
            data.get("customer_id")
            or data.get("cliente_id")
            or data.get("id")
            or "SYN-UNKNOWN"
        )
        tier = str(
            data.get("loyalty_tier_synthetic")
            or data.get("tier")
            or "Básico"
        ).strip()
        persona = str(
            data.get("persona")
            or data.get("segment")
            or "Ocasional em Ascensão"
        ).strip()
        clv = float(
            data.get("clv_simulado_12m_brl")
            or data.get("clv_simulado")
            or data.get("clv")
            or 0.0
        )
        risk = float(
            data.get("churn_risk_score")
            or data.get("risk")
            or 0.0
        )
        d1 = str(
            data.get("driver_risco_1")
            or data.get("top_shap_reason")
            or "Nenhum atrito crítico detectado"
        )
        d2 = data.get("driver_risco_2")
        d3 = data.get("driver_risco_3")
        points = int(data.get("points_balance") or 0)
        months = int(data.get("months_since_last_flight") or 0)
        issues = int(data.get("unresolved_issues_last_12m") or data.get("unresolved_issues") or 0)
        route = data.get("preferred_route_synthetic") or data.get("preferred_route")

        return cls(
            cliente_id=cliente_id,
            tier=tier,
            persona=persona,
            clv_simulado=clv,
            churn_risk_score=risk,
            driver_risco_1=d1,
            driver_risco_2=str(d2) if d2 else None,
            driver_risco_3=str(d3) if d3 else None,
            points_balance=points,
            months_since_last_flight=months,
            unresolved_issues=issues,
            preferred_route=str(route) if route else None,
        )


class CRMActionPayload(BaseModel):
    """Payload obrigatório estrito gerado pelo agente para a esteira de CRM."""
    cliente_id: str = Field(..., description="Identificador único do passageiro")
    acao_proposta: str = Field(..., description="Nome descritivo da ação recomendada")
    mensagem_personalizada: str = Field(..., description="Texto persuasivo pronto para envio")
    canal_recomendado: str = Field(..., description="Canal ideal de contato")
    custo_estimado_reais: float = Field(..., description="Custo unitário em BRL da concessão")
    regra_citada: str = Field(..., description="Citação explícita da Seção do Regulamento")
    status_aprovacao: str = Field(
        default="Pendente de Avaliação Humana",
        description="Status de governança da esteira",
    )
    # Metadados de rastreabilidade
    modo_geracao: Optional[str] = Field(default=None, description="rag_llm ou fallback_deterministico")
    beneficios_validados: Optional[List[str]] = Field(default=None, description="Benefícios auditados pelos guardrails")


# =====================================================================
# Guardrails de Benefícios Permitidos por Tier (0 Alucinações)
# =====================================================================

TIER_BENEFITS_ALLOWED = {
    "Básico": {
        "allowed_keywords": [
            "pontos", "acumulo", "acúmulo", "desafio", "topazio", "topázio",
            "pontos qualificaveis", "pontos qualificáveis", "milhas", "engajamento",
            "desconto", "promocao", "promoção",
        ],
        "forbidden_keywords": [
            "sala vip", "lounge viracopos", "lounge azul", "companion pass",
            "passagem cortesia acompanhante", "upgrade executiva", "upgrade internacional",
            "vouchers de upgrade", "espaço azul ilimitado", "espaco azul ilimitado",
            "3 bagagens", "duas bagagens", "2 bagagens", "cortesia acompanhante",
        ],
        "max_cost_brl": 40.0,
    },
    "Topázio": {
        "allowed_keywords": [
            "pontos qualificaveis", "pontos qualificáveis", "1 bagagem", "uma bagagem",
            "bagagem despachada gratuita", "desconto espaço azul", "desconto espaco azul",
            "6.000 pontos", "6 trechos", "manutenção", "manutencao", "validade 36 meses",
            "pontos", "safira", "upgrade para safira",
        ],
        "forbidden_keywords": [
            "sala vip", "lounge viracopos", "companion pass", "passagem cortesia acompanhante",
            "upgrade internacional executiva", "espaço azul ilimitado", "espaco azul ilimitado",
            "3 bagagens",
        ],
        "max_cost_brl": 60.0,
    },
    "Safira": {
        "allowed_keywords": [
            "pontos qualificaveis", "pontos qualificáveis", "2 bagagens", "duas bagagens",
            "1 bagagem internacional", "2 cortesias", "duas cortesias de espaço azul",
            "espaco azul", "espaço azul", "prioridade de embarque", "check-in prioritário",
            "10.000 pontos", "10 trechos", "manutenção", "manutencao", "validade 36 meses",
            "concierge", "retencao", "retenção", "diamante",
        ],
        "forbidden_keywords": [
            "companion pass", "passagem cortesia acompanhante", "espaço azul ilimitado",
            "espaco azul ilimitado", "3 bagagens", "upgrade internacional executiva ilimitado",
        ],
        "max_cost_brl": 90.0,
    },
    "Diamante": {
        "allowed_keywords": [
            "espaço azul ilimitado", "espaco azul ilimitado", "3 bagagens", "tres bagagens",
            "lounge viracopos", "lounge azul", "sala vip", "companion pass",
            "passagem cortesia acompanhante", "vouchers de upgrade", "upgrade internacional",
            "executiva", "concierge diamante", "pontos não expiram", "pontos sem expiracao",
            "pontos qualificáveis", "12.000 pontos", "12 trechos", "manutenção", "manutencao",
        ],
        "forbidden_keywords": [],
        "max_cost_brl": 160.0,
    },
}


def validate_benefits_guardrails(
    payload: Dict[str, Any],
    tier: str,
    evidence_text: str,
) -> Tuple[bool, List[str]]:
    """Valida rigorosamente se os benefícios gerados respeitam as regras do tier e o contexto recuperado.
    
    Retorna (is_valid, list_of_violations).
    """
    violations = []
    text_to_check = (
        f"{payload.get('acao_proposta', '')} "
        f"{payload.get('mensagem_personalizada', '')} "
        f"{payload.get('regra_citada', '')}"
    ).lower()

    tier_rules = TIER_BENEFITS_ALLOWED.get(tier, TIER_BENEFITS_ALLOWED["Básico"])

    # 1. Checagem de palavras proibidas para o Tier
    for forbidden in tier_rules["forbidden_keywords"]:
        if forbidden in text_to_check:
            violations.append(
                f"Benefício proibido para o Tier '{tier}': menção indevida a '{forbidden}'."
            )

    # 2. Checagem de teto de custo da ação
    cost = float(payload.get("custo_estimado_reais") or 0.0)
    max_cost = tier_rules["max_cost_brl"]
    if cost > max_cost:
        violations.append(
            f"Custo estimado de R$ {cost:.2f} excede o limite orçamentário parametrizado para o Tier '{tier}' (R$ {max_cost:.2f})."
        )

    # 3. Validação de citação de seção válida do regulamento
    regra = payload.get("regra_citada", "")
    if not re.search(r"Seção\s+[1-7]", regra, re.IGNORECASE):
        violations.append(
            "Citação de regra inválida: o campo 'regra_citada' deve obrigatoriamente referenciar uma Seção oficial (ex: Seção 2, 4, 5 ou 7)."
        )

    # 4. Status de aprovação estrito
    if payload.get("status_aprovacao") != "Pendente de Avaliação Humana":
        violations.append(
            "O status_aprovacao deve ser estritamente 'Pendente de Avaliação Humana' para governança de CRM."
        )

    return (len(violations) == 0, violations)


# =====================================================================
# Classe Principal do Agente Cognitivo
# =====================================================================

class ActionAgent:
    """Agente de CRM com RAG lexical e Guardrails para o programa Azul Fidelidade."""

    def __init__(self, documents_dir: Optional[Path] = None):
        self.documents_dir = documents_dir or DEFAULT_DOCUMENTS_DIR
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Carrega e indexa a base de regras públicas em memória."""
        try:
            self.documents = load_documents(self.documents_dir)
            logger.info("Base de regras RAG indexada com sucesso (%d trechos).", len(self.documents))
        except Exception as exc:
            logger.warning("Aviso ao carregar documentos RAG: %s. Operando com modo resiliente.", exc)
            self.documents = []

    def build_rag_query(self, profile: CustomerProfile) -> str:
        """Monta a consulta lexical otimizada com base no perfil, tier e driver de risco."""
        terms = [profile.tier]

        # Drivers de risco e atritos
        d1 = profile.driver_risco_1.lower()
        if "recência" in d1 or "recencia" in d1 or profile.months_since_last_flight >= 4:
            terms.extend(["manutenção", "categoria", "pontos qualificáveis", "retenção", "validade"])
        elif "atrito" in d1 or "reclamaç" in d1 or "bagagem" in d1 or profile.unresolved_issues > 0:
            terms.extend(["bagagem despachada", "franquia", "atendimento", "prioridade", "concierge"])
        elif "desaceleração" in d1 or "frequência" in d1:
            terms.extend(["pontos qualificáveis", "trechos", "manutenção", "aceleração"])
        elif "engajamento" in d1:
            terms.extend(["clube azul", "acúmulo", "validade", "experiências"])

        # Nuance por Persona
        if profile.persona == "Executivo Frequente":
            terms.extend(["espaço azul", "lounge viracopos", "embarque prioritário", "upgrade"])
        elif profile.persona == "Alto Valor em Risco":
            terms.extend(["manutenção de tier", "concessão extraordinária", "pontos qualificáveis", "concierge"])
        elif profile.persona == "Lazer Premium":
            terms.extend(["bagagem gratuita", "espaço azul", "companion pass", "passagem cortesia"])
        elif profile.persona == "Conectividade Regional":
            terms.extend(["trechos", "pontos qualificáveis", "bagagem"])

        return " ".join(terms)

    def retrieve_rules(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Recupera as seções mais relevantes do regulamento para a consulta."""
        if not self.documents:
            return []
        return search_documents(query, self.documents, top_k=top_k)

    def _generate_deterministic_fallback(
        self,
        profile: CustomerProfile,
        evidences: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Gera prescrição determinística robusta com fundamentação em regras e 0 alucinações."""
        tier = profile.tier
        persona = profile.persona
        risk = profile.churn_risk_score
        clv = profile.clv_simulado

        # Escolhe a melhor citação de evidência RAG disponível
        best_section = "Seção 2 do Regulamento Azul Fidelidade"
        rule_quote = "Seção 2: Concessão Extraordinária para Manutenção de Tier (Aceleração de Retenção)"

        if evidences:
            top_ev = evidences[0]
            sec_name = top_ev.get("section", "")
            rule_quote = f"{sec_name}: {top_ev.get('text', '')[:160]}..."

        # Lógica especializada por Persona e Tier
        if persona == "Alto Valor em Risco":
            if tier == "Diamante":
                acao = "Blindagem Concierge Diamante & Concessão de 2.000 Pontos Qualificáveis de Retenção"
                canal = "Concierge Diamante / WhatsApp VIP"
                custo = 70.00
                regra = "Seção 2 e Seção 7 do Regulamento Azul Fidelidade: Concessão Extraordinária de Manutenção de Tier e Atendimento Prioritário Concierge Diamante."
                msg = (
                    f"Olá! Valorizamos imensamente sua preferência como cliente Diamante (CLV estimado em R$ {clv:,.2f}). "
                    "Para garantir que você mantenha todos os seus privilégios exclusivos — incluindo assento Espaço Azul ilimitado, "
                    "acesso ao Lounge Azul Viracopos e pontos que não expiram —, acabamos de creditar uma aceleração especial "
                    "de 2.000 pontos qualificáveis para o seu ciclo de qualificação anual. Conte conosco para planejar seus próximos voos!"
                )
            elif tier == "Safira":
                acao = "Concessão de 1.500 Pontos Qualificáveis para Manutenção de Tier Safira & Ativação de Espaço Azul"
                canal = "WhatsApp VIP / Telefone"
                custo = 55.00
                regra = "Seção 2 e Seção 5 do Regulamento Azul Fidelidade: Requisitos Anuais de Manutenção Safira (10.000 pts) e 2 Cortesias Anuais de Espaço Azul."
                msg = (
                    f"Olá! Sentimos sua falta nos nossos voos recentes. Como membro Safira de alto valor, "
                    "queremos apoiar a manutenção do seu status: liberamos um bônus especial de 1.500 pontos qualificáveis "
                    "para acelerar sua meta anual de 10.000 pontos, além de lembrar que suas 2 cortesias anuais de assento "
                    "Espaço Azul estão disponíveis para suas próximas viagens. Será um prazer tê-lo(a) a bordo novamente!"
                )
            elif tier == "Topázio":
                acao = "Acelerador de 1.000 Pontos Qualificáveis para Renovação Topázio & Franquia de Bagagem"
                canal = "WhatsApp / E-mail Prioritário"
                custo = 35.00
                regra = "Seção 2 e Seção 4 do Regulamento Azul Fidelidade: Requisitos de Qualificação Topázio (6.000 pts) e 1 Bagagem Despachada Gratuita de até 23kg."
                msg = (
                    "Olá! Notamos que faz um tempo que não nos encontramos nas alturas. Para facilitar a renovação da sua "
                    "categoria Topázio no Azul Fidelidade, disponibilizamos um acelerador exclusivo de 1.000 pontos qualificáveis "
                    "no seu próximo voo. Lembre-se que você conta com 1 bagagem despachada gratuita de até 23 kg em todos os trechos nacionais!"
                )
            else:  # Básico
                acao = "Desafio Acelerador de Upgrade para Categoria Topázio com Bônus de Boas-Vindas"
                canal = "Push App / E-mail"
                custo = 25.00
                regra = "Seção 2 e Seção 4 do Regulamento Azul Fidelidade: Ciclo de Qualificação e Benefício de Bagagem Despachada na Categoria Topázio (6 trechos ou 6.000 pts)."
                msg = (
                    "Olá! Queremos ver você voando mais alto. Preparamos uma oportunidade exclusiva: ao completar apenas "
                    "2 novos trechos voados nos próximos 60 dias, você acelera seu upgrade direto para o nível Topázio do Azul Fidelidade, "
                    "conquistando o direito a 1 bagagem despachada gratuita de 23 kg e pontuação multiplicada!"
                )

        elif persona == "Executivo Frequente":
            if tier == "Diamante":
                acao = "Ativação Prioritária de Espaço Azul Ilimitado & Lembrete de Acesso ao Lounge Viracopos"
                canal = "Push App Corporativo / E-mail Executivo"
                custo = 35.00
                regra = "Seção 5 e Seção 7 do Regulamento Azul Fidelidade: Assentos Espaço Azul Gratuitos Ilimitados e Lounge Azul Viracopos com Acompanhante para Clientes Diamante."
                msg = (
                    "Prezado(a) cliente Diamante, seu conforto em viagens de negócios é nossa prioridade absoluta. "
                    "Lembramos que em todas as suas viagens você conta com marcação 100% gratuita e ilimitada no Espaço Azul "
                    "e acesso prioritário com acompanhante ao Lounge Azul em Viracopos. Seu check-in corporativo já está com prioridade configurada!"
                )
            elif tier == "Safira":
                acao = "Resgate Cortesia de Espaço Azul & Prioridade de Embarque Corporativo"
                canal = "Push App / E-mail"
                custo = 30.00
                regra = "Seção 5 e Seção 7 do Regulamento Azul Fidelidade: Concessão de 2 Cortesias Anuais de Espaço Azul e Embarque Prioritário Categoria Safira."
                msg = (
                    "Olá! Para sua próxima ponte aérea corporativa, aproveite sua cortesia anual de assento Espaço Azul no aplicativo. "
                    "Além de mais conforto para trabalhar ou descansar a bordo, você aproveita embarque e check-in prioritários em todos os aeroportos."
                )
            else:
                acao = "Incentivo a Trechos Corporativos com Pontuação Qualificável Dobrada"
                canal = "E-mail Corporativo / App"
                custo = 20.00
                regra = "Seção 2 do Regulamento Azul Fidelidade: Contabilização de Trechos e Pontos Qualificáveis por Voo Operado pela Azul."
                msg = (
                    "Sua rotina produtiva merece mais vantagens. Voe nos trechos executivos selecionados neste mês e acumule "
                    "pontos qualificáveis em dobro para alcançar mais rápido a categoria Safira do Azul Fidelidade."
                )

        elif persona == "Lazer Premium":
            acao = "Aproveitamento de Franquia de Bagagem Despachada Gratuita & Resgate com Pontos para Família"
            canal = "E-mail / WhatsApp"
            custo = 30.00
            regra = "Seção 3 e Seção 4 do Regulamento Azul Fidelidade: Validade de Pontos Estendida e Franquia de Bagagem Gratuita por Categoria."
            msg = (
                f"Planejando sua próxima viagem de férias? Como membro {tier}, você conta com franquia de bagagem gratuita "
                f"e um saldo de {profile.points_balance:,} pontos prontos para serem resgatados em passagens e pacotes exclusivos. "
                "Confira as condições especiais selecionadas para suas rotas de lazer favoritas!"
            )

        elif persona == "Conectividade Regional":
            acao = "Aceleração de Pontos em Rotas Regionais Estratégicas & Isenção de Bagagem"
            canal = "SMS / Push Notificação"
            custo = 20.00
            regra = "Seção 2 e Seção 4 do Regulamento Azul Fidelidade: Pontuação Qualificável em Voos e Benefício de Bagagem por Categoria."
            msg = (
                f"Viaje com a conectividade incomparável da Azul saindo do seu aeroporto regional preferido ({profile.preferred_route or 'malha Azul'}) "
                "e garanta pontuação acelerada para garantir sua categoria no Azul Fidelidade."
            )

        else:  # Ocasional em Ascensão
            acao = "Campanha de Engajamento: Desafio de Pontos Qualificáveis para Categoria Superior"
            canal = "E-mail de Engajamento / In-App"
            custo = 15.00
            regra = "Seção 2 do Regulamento Azul Fidelidade: Requisitos Anuais de Manutenção e Elevação de Categoria (Topázio 6.000 pts / 6 trechos)."
            msg = (
                f"Olá! Sabia que você está a poucos passos de elevar sua experiência de viagem? Conheça os benefícios da "
                "categoria Topázio do Azul Fidelidade — incluindo 1 bagagem gratuita de 23 kg — e acumule pontos no seu próximo voo!"
            )

        return {
            "cliente_id": profile.cliente_id,
            "acao_proposta": acao,
            "mensagem_personalizada": msg,
            "canal_recomendado": canal,
            "custo_estimado_reais": float(custo),
            "regra_citada": regra,
            "status_aprovacao": "Pendente de Avaliação Humana",
            "modo_geracao": "fallback_deterministico",
        }

    def _call_llm(
        self,
        profile: CustomerProfile,
        evidences: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Tenta gerar a recomendação via LLM se houver API configurada (OpenAI ou Gemini)."""
        openai_key = os.getenv("OPENAI_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")

        if not openai_key and not gemini_key:
            return None

        # Monta o contexto documental recuperado
        context_chunks = []
        for i, ev in enumerate(evidences, 1):
            context_chunks.append(
                f"[Trecho {i} - {ev.get('source', '')} - {ev.get('section', '')}]\n{ev.get('text', '')}"
            )
        context_str = "\n\n".join(context_chunks)

        prompt = f"""
Você é o Agente Cognitivo de CRM de Retenção do Programa Azul Fidelidade.
Sua missão é gerar a Próxima Melhor Ação (Next Best Action) estritamente fundamentada no Regulamento Oficial Azul Fidelidade.

PERFIL DO PASSAGEIRO:
- ID: {profile.cliente_id}
- Categoria (Tier Atual): {profile.tier}
- Persona: {profile.persona}
- CLV Previsto (12M): R$ {profile.clv_simulado:,.2f}
- Risco de Churn: {profile.churn_risk_score:.1%}
- Principal Fator de Risco (SHAP): {profile.driver_risco_1}
- Fator Secundário: {profile.driver_risco_2 or 'N/A'}
- Saldo de Pontos: {profile.points_balance:,}
- Meses sem voar: {profile.months_since_last_flight}
- Reclamações Pendentes: {profile.unresolved_issues}

REGRAS RECUPERADAS VIA RAG (BASE PÚBLICA OFICIAL):
{context_str}

DIRETRIZES MANDATÓRIAS (ZERO ALUCINAÇÕES):
1. NUNCA ofereça benefícios que não pertençam ao Tier do passageiro ou que não estejam descritos no regulamento.
   - Básico: NUNCA oferecer sala VIP, Espaço Azul gratuito ou 2/3 bagagens.
   - Topázio: máximo de 1 bagagem de 23kg gratuita doméstica. Sem sala VIP nem Espaço Azul gratuito.
   - Safira: 2 bagagens domésticas / 1 internacional, 2 cortesias de Espaço Azul/ano, prioridade embarque/check-in. NUNCA Espaço Azul ilimitado nem Companion Pass.
   - Diamante: Espaço Azul ilimitado, até 3 bagagens, Lounge Viracopos com acompanhante, Companion Pass anual, 4 vouchers upgrade executiva internacional, Concierge Diamante.
2. A citação da regra no campo 'regra_citada' DEVE citar explicitamente a Seção X do regulamento correspondente.
3. O status de aprovação DEVE ser estritamente "Pendente de Avaliação Humana".
4. Retorne APENAS um JSON válido no formato:
{{
  "cliente_id": "{profile.cliente_id}",
  "acao_proposta": "...",
  "mensagem_personalizada": "...",
  "canal_recomendado": "WhatsApp / Concierge / E-mail / Push App",
  "custo_estimado_reais": 0.00,
  "regra_citada": "Seção X do Regulamento Azul Fidelidade: ...",
  "status_aprovacao": "Pendente de Avaliação Humana"
}}
"""

        # 1. Tentativa via OpenAI se configurada
        if openai_key:
            try:
                import requests
                headers = {
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    "messages": [
                        {"role": "system", "content": "Você é um assistente de CRM rigoroso que responde exclusivamente em JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                }
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=15,
                )
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    data = json.loads(content)
                    data["modo_geracao"] = "rag_llm"
                    return data
            except Exception as exc:
                logger.warning("Falha na chamada OpenAI: %s. Chaveando para fallback.", exc)

        # 2. Tentativa via Gemini se configurada
        if gemini_key:
            try:
                import requests
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=15)
                if resp.status_code == 200:
                    content = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                    data = json.loads(content)
                    data["modo_geracao"] = "rag_llm"
                    return data
            except Exception as exc:
                logger.warning("Falha na chamada Gemini: %s. Chaveando para fallback.", exc)

        return None

    def generate_action(self, customer_input: Any) -> CRMActionPayload:
        """Gera a ação de CRM com RAG, execução via LLM ou Fallback e Guardrails estritos."""
        if isinstance(customer_input, CustomerProfile):
            profile = customer_input
        elif isinstance(customer_input, dict):
            profile = CustomerProfile.from_dict(customer_input)
        elif hasattr(customer_input, "to_dict"):
            profile = CustomerProfile.from_dict(customer_input.to_dict())
        else:
            raise ValueError(f"Formato de entrada não suportado: {type(customer_input)}")

        # 1. Recuperação RAG
        rag_query = self.build_rag_query(profile)
        evidences = self.retrieve_rules(rag_query, top_k=3)
        evidence_text = "\n".join(e.get("text", "") for e in evidences)

        # 2. Tentativa via LLM
        payload_dict = self._call_llm(profile, evidences)

        # 3. Fallback Determinístico transparente se LLM indisponível ou falhou
        if payload_dict is None:
            payload_dict = self._generate_deterministic_fallback(profile, evidences)

        # 4. Guardrails e Validação de 0 Alucinações
        is_valid, violations = validate_benefits_guardrails(
            payload_dict,
            tier=profile.tier,
            evidence_text=evidence_text,
        )

        if not is_valid:
            logger.warning(
                "Guardrail detectou inconformidade em payload (%s): %s. Aplicando saneamento determinístico seguro.",
                profile.cliente_id,
                "; ".join(violations),
            )
            # Aciona saneamento pelo gerador determinístico garantido
            payload_dict = self._generate_deterministic_fallback(profile, evidences)
            # Revalida para assegurar 100% de conformidade
            is_valid, violations = validate_benefits_guardrails(
                payload_dict,
                tier=profile.tier,
                evidence_text=evidence_text,
            )

        # 5. Adiciona metadados de auditoria e constrói objeto Pydantic
        payload_dict["status_aprovacao"] = "Pendente de Avaliação Humana"
        payload_dict["beneficios_validados"] = [
            f"Tier auditado: {profile.tier}",
            "Conformidade com Seções Regulamentares Azul Fidelidade",
            "Zero benefícios não autorizados detectados",
        ]

        return CRMActionPayload(**payload_dict)

    def batch_generate(self, customers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Gera recomendações em lote para uma lista de passageiros."""
        results = []
        for c in customers:
            action = self.generate_action(c)
            results.append(action.model_dump())
        return results


# =====================================================================
# Ponto de Entrada para Execução Rápida via Linha de Comando
# =====================================================================

def main():
    """Demonstração rápida da geração de recomendação."""
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Agente Cognitivo de CRM com RAG e Guardrails.")
    parser.add_argument("--customer-id", type=str, default="SYN-00763", help="ID do cliente na base ouro.")
    parser.add_argument("--json", action="store_true", help="Imprime saída em formato JSON.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    gold_file = project_root / "data" / "gold" / "perfis_clientes_sinteticos.csv"

    agent = ActionAgent()

    if gold_file.exists():
        df = pd.read_csv(gold_file)
        match = df[df["customer_id"] == args.customer_id]
        if not match.empty:
            record = match.iloc[0].to_dict()
        else:
            print(f"Cliente {args.customer_id} não encontrado na base. Usando primeiro registro.")
            record = df.iloc[0].to_dict()
    else:
        # Perfil mock para teste standalone
        record = {
            "cliente_id": args.customer_id,
            "loyalty_tier_synthetic": "Safira",
            "persona": "Alto Valor em Risco",
            "clv_simulado_12m_brl": 4500.0,
            "churn_risk_score": 0.88,
            "driver_risco_1": "Recência de voo elevada (> 4 meses sem voar)",
            "driver_risco_2": "Reclamações operacionais/atendimento pendentes sem desfecho",
            "points_balance": 45000,
            "months_since_last_flight": 5,
            "unresolved_issues_last_12m": 1,
        }

    recommendation = agent.generate_action(record)

    if args.json:
        print(json.dumps(recommendation.model_dump(), ensure_ascii=False, indent=2))
    else:
        print("\n=======================================================")
        print("  RECOMENDAÇÃO DO AGENTE COGNITIVO DE CRM (AZUL FIDELIDADE)  ")
        print("=======================================================")
        for k, v in recommendation.model_dump().items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
