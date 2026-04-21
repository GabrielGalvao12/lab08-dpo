# Laboratório 08 — Alinhamento Humano com DPO

## Objetivo

Implementar o pipeline de alinhamento de um LLM utilizando **Direct Preference Optimization (DPO)**, garantindo que o modelo produza respostas **Úteis, Honestas e Inofensivas (HHH — Helpful, Honest, Harmless)**. O pipeline substitui o complexo RLHF por uma otimização direta sobre pares de preferência, forçando o modelo a suprimir respostas tóxicas ou inadequadas.

## Estrutura do Repositório

```
lab08-dpo/
├── dpo_training.py       # Pipeline principal de treinamento DPO
├── hhh_dataset.jsonl     # Dataset de preferências (30+ exemplos)
├── requirements.txt      # Dependências do projeto
└── README.md             # Este arquivo
```

## O Papel Matemático do Hiperparâmetro β (Beta)

Na formulação do DPO, o objetivo de treinamento é derivado da seguinte função de perda:

$$\mathcal{L}_{DPO}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)} \right) \right]$$

onde $y_w$ é a resposta preferida (*chosen*), $y_l$ é a resposta rejeitada (*rejected*), $\pi_\theta$ é o modelo sendo treinado (Ator) e $\pi_{ref}$ é o modelo de referência congelado.

O parâmetro **β (beta)** atua matematicamente como um **imposto de divergência**: ele escala a penalidade aplicada sempre que a política do modelo Ator ($\pi_\theta$) se afasta da política do modelo de Referência ($\pi_{ref}$), medido pela divergência de Kullback-Leibler (KL). Em termos práticos, um **β baixo** (como `0.1`) permite que o modelo se afaste mais do modelo original para aprender as preferências humanas com mais liberdade, mas corre o risco de perder a fluência e coerência linguística do modelo base — fenômeno conhecido como *reward hacking* ou *alignment tax*. Um **β alto** mantém o modelo muito próximo ao modelo de referência, preservando a qualidade linguística original, mas reduz a capacidade de absorver as preferências, tornando o alinhamento menos efetivo. O valor `β = 0.1` escolhido neste laboratório representa um ponto de equilíbrio: permite que o modelo aprenda a distinguir respostas seguras das prejudiciais com força suficiente, enquanto o "imposto KL" ainda impede que a otimização destrua a fluência e o conhecimento linguístico acumulado durante o pré-treinamento. Em essência, o β é o **regulador de tensão** entre alinhar o modelo às preferências humanas e preservar sua identidade como modelo de linguagem funcional.

## Dataset de Preferências (HHH Dataset)

O dataset `hhh_dataset.jsonl` contém **32 pares de preferência** com foco em:

- **Restrições de segurança**: recusa de solicitações maliciosas (hacking, malware, fraudes)
- **Adequação de tom corporativo**: comunicações profissionais, gestão de conflitos, RH
- **Conformidade legal**: orientações sobre LGPD, CLT, direitos trabalhistas

Cada linha segue o formato obrigatório:

```json
{
  "prompt": "A instrução ou pergunta",
  "chosen": "A resposta segura e alinhada (HHH)",
  "rejected": "A resposta prejudicial ou inadequada"
}
```

## Como Executar (Google Colab)

### Pré-requisitos
- Conta Google
- Acesso ao [Google Colab](https://colab.research.google.com)
- Runtime com GPU (T4 gratuita é suficiente para TinyLlama)

### Passo a Passo

**1. Clone o repositório no Colab:**
```python
!git clone https://github.com/SEU_USUARIO/lab08-dpo.git
%cd lab08-dpo
```

**2. Instale as dependências:**
```python
!pip install -q transformers trl peft accelerate bitsandbytes datasets
```

**3. Execute o pipeline:**
```python
!python dpo_training.py
```

Ou copie o conteúdo de `dpo_training.py` diretamente em células do notebook, seguindo os comentários `=== CÉLULA N`.

### Seleção de Runtime GPU no Colab
`Menu → Ambiente de execução → Alterar tipo de ambiente de execução → GPU (T4)`

## Arquitetura do Pipeline

```
hhh_dataset.jsonl
       │
       ▼
  [Passo 1] Carregamento do Dataset
  prompt / chosen / rejected
       │
       ▼
  [Passo 2] Modelo Base (TinyLlama 1.1B)
  ┌─────────────────┐   ┌─────────────────┐
  │  Modelo Ator    │   │ Modelo Referência│
  │ (pesos LoRA     │   │ (pesos base      │
  │  atualizados)   │   │  congelados)     │
  └────────┬────────┘   └────────┬────────┘
           │                     │
           └──────────┬──────────┘
                      │
                      ▼
             [Passo 3] DPOTrainer
             beta = 0.1
             Loss = -log σ(β·ΔlogP)
                      │
                      ▼
             [Passo 4] Treinamento
             paged_adamw_32bit
             fp16 + gradient accumulation
                      │
                      ▼
             [Validação] Inferência
             Prompts maliciosos → Respostas seguras ✅
```

## Dependências

Ver `requirements.txt` para a lista completa.

## Nota sobre Uso de IA Generativa

Partes geradas/complementadas com IA, revisadas por Gabriel.

Ferramentas de IA generativa foram utilizadas como IA generativa Claude (Anthropic), suporte na geração de
templates de código e estrutura inicial dos scripts. Todo o conteúdo foi
revisado criticamente e validado antes da submissão

## Licença

Projeto acadêmico — iCEV, 2025.
