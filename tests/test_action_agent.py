"""Testes automatizados e validação do Agente Cognitivo de CRM com RAG e Guardrails."""

import json
import unittest
from pathlib import Path

from src.agent.action_agent import (
    ActionAgent,
    CRMActionPayload,
    CustomerProfile,
    validate_benefits_guardrails,
)


class ActionAgentTests(unittest.TestCase):
    """Bateria de testes e conformidade de regras de negócio para o ActionAgent."""

    @classmethod
    def setUpClass(cls):
        cls.agent = ActionAgent()

    def test_high_value_at_risk_safira(self):
        """Valida geração para cliente Safira com Persona 'Alto Valor em Risco'."""
        customer = {
            "customer_id": "SYN-00763",
            "loyalty_tier_synthetic": "Safira",
            "persona": "Alto Valor em Risco",
            "clv_simulado_12m_brl": 3902.76,
            "churn_risk_score": 0.9048,
            "driver_risco_1": "Recência de voo elevada (> 4 meses sem voar)",
            "driver_risco_2": "Reclamações operacionais/atendimento pendentes sem desfecho",
            "points_balance": 42000,
            "months_since_last_flight": 5,
            "unresolved_issues_last_12m": 1,
        }

        action = self.agent.generate_action(customer)
        data = action.model_dump()

        # Validação estrutural do payload JSON
        self.assertEqual(data["cliente_id"], "SYN-00763")
        self.assertIn("Safira", data["acao_proposta"])
        self.assertTrue(len(data["mensagem_personalizada"]) > 20)
        self.assertIn("WhatsApp", data["canal_recomendado"])
        self.assertGreater(data["custo_estimado_reais"], 0.0)
        self.assertIn("Seção", data["regra_citada"])
        self.assertEqual(data["status_aprovacao"], "Pendente de Avaliação Humana")

        # Zero alucinações para Safira: NÃO pode ter Companion Pass nem Espaço Azul ilimitado
        msg_lower = data["mensagem_personalizada"].lower()
        self.assertNotIn("companion pass", msg_lower)
        self.assertNotIn("espaço azul ilimitado", msg_lower)
        self.assertNotIn("3 bagagens", msg_lower)

    def test_frequent_executive_diamante(self):
        """Valida geração para cliente Diamante com Persona 'Executivo Frequente'."""
        customer = {
            "customer_id": "SYN-00003",
            "loyalty_tier_synthetic": "Diamante",
            "persona": "Executivo Frequente",
            "clv_simulado_12m_brl": 83063.21,
            "churn_risk_score": 0.0012,
            "driver_risco_1": "Forte desaceleração na frequência de trechos voados",
            "driver_risco_2": "Variação na margem operacional das rotas utilizadas",
            "points_balance": 217971,
            "months_since_last_flight": 0,
            "unresolved_issues_last_12m": 0,
        }

        action = self.agent.generate_action(customer)
        data = action.model_dump()

        self.assertEqual(data["cliente_id"], "SYN-00003")
        self.assertIn("Espaço Azul", data["acao_proposta"])
        self.assertIn("Seção 5", data["regra_citada"])
        self.assertIn("Seção 7", data["regra_citada"])
        self.assertEqual(data["status_aprovacao"], "Pendente de Avaliação Humana")
        self.assertGreater(data["custo_estimado_reais"], 0.0)

    def test_basic_customer_zero_hallucinations(self):
        """Garante que cliente da categoria Básico NUNCA receba benefícios exclusivos de tiers superiores."""
        customer = {
            "customer_id": "SYN-00001",
            "loyalty_tier_synthetic": "Básico",
            "persona": "Ocasional em Ascensão",
            "clv_simulado_12m_brl": 367.18,
            "churn_risk_score": 0.7029,
            "driver_risco_1": "Queda acentuada de engajamento no aplicativo/portal",
            "points_balance": 5455,
            "months_since_last_flight": 1,
            "unresolved_issues_last_12m": 0,
        }

        action = self.agent.generate_action(customer)
        data = action.model_dump()

        combined_text = f"{data['acao_proposta']} {data['mensagem_personalizada']}".lower()

        # Proibidos para Básico:
        self.assertNotIn("sala vip", combined_text)
        self.assertNotIn("lounge viracopos", combined_text)
        self.assertNotIn("companion pass", combined_text)
        self.assertNotIn("espaço azul ilimitado", combined_text)
        self.assertNotIn("upgrade executiva", combined_text)

    def test_guardrails_prevent_unauthorized_benefits(self):
        """Valida que o Guardrail reprova explicitamente propostas que contenham benefícios indevidos."""
        # Tentativa forçada de dar Sala VIP e Companion Pass para cliente Básico
        hallucinated_payload = {
            "cliente_id": "SYN-TEST-FAIL",
            "acao_proposta": "Concessão de Sala VIP e Companion Pass Cortesia",
            "mensagem_personalizada": "Você ganhou acesso à Sala VIP Lounge Viracopos e passagem cortesia acompanhante!",
            "canal_recomendado": "WhatsApp",
            "custo_estimado_reais": 250.0,
            "regra_citada": "Seção 7 do Regulamento",
            "status_aprovacao": "Pendente de Avaliação Humana",
        }

        is_valid, violations = validate_benefits_guardrails(
            payload=hallucinated_payload,
            tier="Básico",
            evidence_text="",
        )

        self.assertFalse(is_valid)
        self.assertTrue(len(violations) >= 2)
        self.assertTrue(any("sala vip" in v.lower() for v in violations))
        self.assertTrue(any("companion pass" in v.lower() for v in violations))
        self.assertTrue(any("excede o limite orçamentário" in v.lower() for v in violations))

    def test_pydantic_schema_compliance(self):
        """Garante validação estrita dos campos obrigatórios da saída em JSON."""
        customer = {
            "customer_id": "SYN-00207",
            "loyalty_tier_synthetic": "Topázio",
            "persona": "Alto Valor em Risco",
            "clv_simulado_12m_brl": 2464.73,
            "churn_risk_score": 0.8015,
            "driver_risco_1": "Recência de voo elevada (> 4 meses sem voar)",
        }
        action = self.agent.generate_action(customer)
        self.assertIsInstance(action, CRMActionPayload)
        payload_json = action.model_dump_json()
        parsed = json.loads(payload_json)

        required_keys = {
            "cliente_id",
            "acao_proposta",
            "mensagem_personalizada",
            "canal_recomendado",
            "custo_estimado_reais",
            "regra_citada",
            "status_aprovacao",
        }
        self.assertTrue(required_keys.issubset(parsed.keys()))


if __name__ == "__main__":
    unittest.main()
