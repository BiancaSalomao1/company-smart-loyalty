# Roteiro de pitch — Company Smart Loyalty

**Duração alvo:** 5 minutos.  
**Regra de apresentação:** chamar os dados de reclamação de públicos, os perfis de clientes de sintéticos e o ROI de simulado. Não apresentar nenhum score como evidência de clientes reais.

## 0:00–0:35 — Problema

> Companhias aéreas recebem sinais de atrito em diversas fontes, mas esses sinais raramente chegam organizados ao time que precisa agir. O resultado é uma reação genérica: a empresa sabe que há reclamações, mas não sabe quais experiências investigar, qual impacto acompanhar ou como preparar ações de relacionamento responsáveis.

> O Company Smart Loyalty é um MVP de inteligência de CRM que organiza esses sinais para transformar reclamações públicas, regras documentadas de fidelidade e perfis sintéticos em prioridades de atuação explicáveis.

**Tela:** visão executiva do dashboard, com o aviso de limitações visível.

## 0:35–1:15 — Evidência e oportunidade

> Começamos com dados públicos de reclamações de 2025. O painel mostra volume, taxa de não resolução, tempo médio de resposta, nota do consumidor e os problemas mais críticos. Esses dados não identificam clientes individuais e não comprovam churn; eles indicam onde investigar e quais hipóteses testar.

> Para comparar alternativas de atuação, incluímos cenários de ROI explicitamente simulados. Assim, o gestor consegue discutir custo, benefício e premissas — sem confundir projeção com resultado observado.

**Tela:** evolução de reclamações, radar de problemas e cenários de ROI.

## 1:15–2:15 — Demonstração do fluxo de CRM

> Como dados de clientes não estão disponíveis no desafio, criamos uma base determinística de passageiros sintéticos. Cada perfil recebe frequência, recência, ticket médio, pontos, engajamento e ocorrências de atendimento fictícios.

> Sobre essa base, o MVP calcula CLV simulado, um score de risco demonstrativo, um segmento e uma próxima ação. Vamos selecionar um perfil sintético de alto risco. O painel explica a prioridade, mostra a ação recomendada e deixa claro que ela depende de validação de elegibilidade e de dados reais antes de qualquer contato.

**Tela:** seção “Clientes sintéticos: valor, risco e ação”; selecionar um perfil de risco elevado.

## 2:15–3:10 — IA com evidências, não com promessas

> A camada de conhecimento consulta a documentação do programa de fidelidade e recupera os trechos mais relevantes. A resposta é estruturada em fatos documentados, hipóteses, recomendações, limitações e fontes.

> A LLM não calcula indicadores nem decide benefícios. Os cálculos continuam em código, e a IA apenas organiza a interpretação com evidências citadas. Sem evidência suficiente, o sistema deve dizer isso.

**Tela:** diagnóstico documental; se a chave de API não estiver configurada, demonstrar a recuperação de evidências em vez de prometer geração em tempo real.

## 3:10–3:55 — Diferencial e governança

> O diferencial não é prometer prever o comportamento de passageiros sem dados. É entregar uma arquitetura pronta para validação: dados públicos e sintéticos separados, regras documentadas, recomendações explicáveis e humano no circuito antes de qualquer ação.

> Com dados autorizados, a evolução prevista é validar o modelo temporalmente, analisar viés, calibrar as premissas e medir resultado com grupos de controle.

**Tela:** arquitetura Bronze–Silver–Gold–DuckDB–Streamlit e documentação de dados sintéticos.

## 3:55–4:35 — Valor e próximos passos

> Hoje, o MVP ajuda a priorizar investigações de experiência e a testar ações de relacionamento com cenários transparentes. Amanhã, com dados autorizados, ele pode apoiar uma carteira real de retenção: identificar receita em risco, explicar os sinais e medir quais ações funcionam.

> Os próximos passos são: conectar dados aprovados de voo, fidelidade e transações; validar o modelo contra um período futuro; testar campanhas controladas; e integrar a recomendação aprovada ao CRM.

**Tela:** lista curta de próximos passos e ressalva de validação.

## 4:35–5:00 — Fechamento

> Company Smart Loyalty transforma sinais dispersos em uma conversa mais responsável sobre retenção: o que observamos, o que ainda é hipótese e qual ação vale testar primeiro. Em vez de automatizar promessas, entregamos decisões rastreáveis e prontas para validação.

> Obrigado.

## Preparação antes de gravar

1. Execute a geração dos dados sintéticos, a inteligência de clientes e a reconstrução do DuckDB conforme o README.
2. Abra o dashboard e deixe pré-selecionado um perfil sintético de alto risco.
3. Se houver chave OpenAI, valide uma pergunta documental previamente; se não houver, não inclua uma geração de LLM ao vivo.
4. Use apenas os valores mostrados no dashboard e declare a natureza pública, sintética ou simulada de cada resultado.
5. Grave uma versão de até cinco minutos e mantenha uma gravação de contingência sem dependência de rede.
