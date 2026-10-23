# Dados sintéticos e modelos demonstrativos

`src/synthetic/generate_passengers.py` cria passageiros inteiramente fictícios com uma semente fixa. Eles não representam pessoas, clientes, transações ou comportamento da Azul.

Os campos incluem frequência de voos, recência, ticket médio, pontos, ocorrências de atendimento e engajamento. `churned_synthetic` é um alvo probabilístico criado apenas para permitir uma demonstração de avaliação de modelo.

`src/analysis/build_customer_intelligence.py` gera:

- CLV simulado de 12 meses: frequência sintética × ticket médio sintético × fator de margem demonstrativo;
- score de risco: baseline de regressão logística treinado e avaliado exclusivamente nesse conjunto artificial;
- segmentos explicáveis e ações sugeridas por regras.

As métricas, scores, segmentos, ações e recomendações resultantes não devem ser usados para decisão operacional, concessão de benefício ou tratamento de um passageiro real. Antes de qualquer uso em produção são necessários dados autorizados, validação temporal, revisão de viés, governança e aprovação de privacidade.
