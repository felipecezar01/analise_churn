# 🤖 Motor de Retenção Ativa com IA Agêntica

> Preveja quem vai embora — e aja antes que seja tarde.

Um sistema de **Machine Learning + IA Generativa** que identifica clientes em risco de churn e gera, automaticamente, estratégias de retenção personalizadas com aprovação humana no loop.

---

## 🎯 Objetivo do Projeto

Bancos e fintechs perdem bilhões por ano por não agirem a tempo quando um cliente decide ir embora. Este projeto resolve exatamente isso.

**O fluxo de valor do sistema:**

```
10.000 clientes históricos
        │
        ▼
 Random Forest (scikit-learn)
 aprende quem cancela e por quê
        │
        ▼
 Novo cliente entra no radar
 → predict_proba() calcula risco
        │
    Churn > 70%?
        │
        ▼
 Gemini identifica o fator crítico
 e redige um e-mail de retenção
 personalizado e empático
        │
        ▼
 Gerente aprova, rejeita ou refina
 com linguagem natural (Human-in-the-loop)
```

Este projeto demonstra na prática a integração entre **ML clássico** (interpretável, auditável) e **IA Generativa** (criativa, conversacional) — uma combinação cada vez mais exigida pelo mercado.

---

## 🛠️ Tecnologias

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.11+ |
| Machine Learning | Scikit-Learn (Random Forest, Pipeline, ColumnTransformer) |
| Manipulação de Dados | Pandas, NumPy |
| IA Generativa | Google Generative AI SDK (Gemini) |
| Persistência de Modelo | Joblib |
| Segurança de Credenciais | python-dotenv |

---

## 📁 Estrutura do Projeto

```
analise_churn/
├── data/
│   └── churn_pred.csv          ← Dataset com 10.000 clientes
├── models/                     ← Gerado após rodar train.py (ignorado pelo git)
│   ├── pipeline.joblib
│   └── feature_importance.joblib
├── src/
│   ├── train.py                ← Treinamento, avaliação e salvamento do modelo
│   └── main.py                 ← Aplicação principal com integração Gemini
├── .env                        ← Sua chave de API (NUNCA commitar)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Como Executar

### Pré-requisitos

- Python 3.11 ou superior
- Uma chave de API do Google Gemini (gratuita em [aistudio.google.com](https://aistudio.google.com/app/apikey))

---

### Passo 1 — Clonar o repositório

```bash
git clone https://github.com/felipecezar01/analise_churn.git
cd analise_churn
```

---

### Passo 2 — Criar e ativar o ambiente virtual

```bash
# Criar
python3 -m venv venv

# Ativar (Linux / macOS)
source venv/bin/activate

# Ativar (Windows)
venv\Scripts\activate
```

---

### Passo 3 — Instalar as dependências

```bash
pip install -r requirements.txt
```

---

### Passo 4 — Configurar a chave de API

Crie um arquivo `.env` na raiz do projeto:

```bash
# Linux / macOS
touch .env
```

Abra o arquivo e adicione sua chave:

```
GEMINI_API_KEY=sua_chave_real_aqui
```

> ⚠️ **Importante:** o arquivo `.env` já está no `.gitignore` e **nunca deve ser commitado**. Sua chave fica protegida localmente.

---

### Passo 5 — Treinar o modelo

```bash
python src/train.py
```

Você verá no terminal as métricas de avaliação (Acurácia, F1-Score) e o ranking das variáveis mais importantes para a previsão. O modelo treinado será salvo automaticamente na pasta `/models`.

---

### Passo 6 — Executar a aplicação

```bash
python src/main.py
```

O sistema apresentará uma introdução sobre o modelo e 3 perfis de clientes para simular. Ao selecionar o cliente em alto risco, o Gemini será acionado para gerar uma proposta de retenção, e você poderá **aprovar, rejeitar ou refinar** o texto gerado com linguagem natural.

---

## 📊 Dicionário de Dados

| Coluna | Tipo | Descrição |
|---|---|---|
| `customer_id` | int | Identificador único — descartado no treinamento |
| `credit_score` | int | Pontuação de crédito (350–850). Score baixo → maior insatisfação |
| `country` | string | País do cliente (France, Germany, Spain) |
| `gender` | string | Gênero do cliente |
| `age` | int | Idade em anos — clientes mais velhos apresentam maior churn neste dataset |
| `tenure` | int | Anos de relacionamento com o banco (0–10) |
| `balance` | float | Saldo médio na conta. Saldo zero = forte sinal de desengajamento |
| `products_number` | int | Nº de produtos contratados (1–4) |
| `credit_card` | int | Possui cartão de crédito? (1 = sim) |
| `active_member` | int | Cliente ativo? (1 = sim). Inatividade é um dos maiores preditores de churn |
| `estimated_salary` | float | Renda anual estimada — calibra o perfil socioeconômico |
| `churn` | int | **Variável alvo.** 1 = cancelou, 0 = permaneceu |

---

## 💡 Destaques Técnicos

- **Pipeline sklearn completo** salvo com joblib: garante que os dados de inferência passem pelas mesmas transformações do treino, eliminando data leakage.
- **ColumnTransformer** com `StandardScaler` (numéricas) + `OneHotEncoder` (categóricas): pré-processamento modular e extensível.
- **Sessão de chat contínua com o Gemini**: o histórico da conversa é mantido automaticamente pelo SDK, permitindo refinamentos iterativos do e-mail sem perder contexto.
- **Human-in-the-loop**: nenhuma ação é tomada sem aprovação explícita do usuário — padrão de IA responsável.

---

## 📄 Licença

MIT License — sinta-se livre para usar, modificar e distribuir.
