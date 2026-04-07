# 🏦 Churn Predictor & AI Retention Agent

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-F7931E?logo=scikit-learn&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google_Gemini-Generative_AI-8E75B2?logo=google&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?logo=pandas&logoColor=white)

> Preveja quem vai embora — e aja antes que seja tarde.

Um sistema híbrido de **Machine Learning Clássico + IA Generativa** que identifica clientes bancários em risco de evasão (Churn) e gera, automaticamente, estratégias de retenção personalizadas com aprovação humana no loop (Human-in-the-loop).

---

## 🎯 O Que é Este Projeto?

Bancos e fintechs perdem bilhões anualmente por não agirem a tempo quando um cliente dá sinais de que vai encerrar sua conta. Este projeto resolve esse gap de comunicação combinando a precisão matemática da estatística com a fluidez da Inteligência Artificial.

**A Origem dos Dados:**
O sistema foi treinado utilizando o clássico **[Bank Customer Churn Dataset](https://www.kaggle.com)** do Kaggle. Trata-se de uma amostra histórica de 10.000 clientes de um banco europeu (operando na França, Alemanha e Espanha), contendo dados transacionais, demográficos e o status final de retenção do cliente.

**O Fluxo de Valor do Sistema:**

```text
10.000 clientes históricos do Banco Europeu
        │
        ▼
 Random Forest (scikit-learn)
 aprende os padrões de quem cancela a conta
        │
        ▼
 Novo cliente entra no radar do sistema
 → predict_proba() calcula a % de risco
        │
    Risco de Churn > 70%?
        │
        ▼
 A API do Gemini entra em ação: identifica o fator 
 crítico (ex: saldo zerado, pouca interação) e redige 
 um e-mail de retenção estratégico e empático.
        │
        ▼
 O Gerente do banco aprova, rejeita ou refina
 o e-mail via chat (Human-in-the-loop) antes do envio.
```

Este projeto demonstra na prática a integração exigida pelo mercado atual: **ML clássico** (interpretável para tomada de decisão) rodando em conjunto com **IA Generativa** (criativa, para escalar a comunicação).

---

## 🛠️ Stack Tecnológico

| Camada | Tecnologia |
|---|---|
| **Linguagem** | Python 3.11+ |
| **Machine Learning** | Scikit-Learn (Random Forest, Pipeline, ColumnTransformer) |
| **Engenharia de Dados** | Pandas, NumPy |
| **IA Generativa** | Google Generative AI SDK (Gemini API) |
| **Persistência de Modelo** | Joblib |
| **Segurança** | python-dotenv |

---

## ⚙️ Como Executar o Projeto

### Pré-requisitos
- Python 3.11 ou superior.
- Uma chave de API gratuita do Google Gemini (obtenha em [aistudio.google.com](https://aistudio.google.com/app/apikey)).

### Passo a Passo

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/felipecezar01/analise_churn.git](https://github.com/felipecezar01/analise_churn.git)
   cd analise_churn
   ```

2. **Crie e ative o ambiente virtual:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # ou: venv\Scripts\activate no Windows
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure a sua Chave de API:**
   Crie um arquivo `.env` na raiz do projeto e adicione sua chave. *(Nota: o arquivo .env já está no .gitignore para garantir a segurança da sua credencial).*
   ```env
   GEMINI_API_KEY="sua_chave_real_aqui"
   ```

5. **Treine o Modelo de Machine Learning:**
   ```bash
   python src/train.py
   ```
   *Este comando processará o CSV do banco, exibirá as métricas (Acurácia, F1-Score) e salvará o pipeline treinado na pasta `/models`.*

6. **Execute o Agente de Retenção:**
   ```bash
   python src/main.py
   ```
   *A interface no terminal permitirá que você simule clientes. Se o cliente estiver em alto risco, o Gemini irá gerar o e-mail de retenção e você poderá interagir com a IA para refinar o texto.*

---

## 📁 Estrutura do Projeto

```text
analise_churn/
├── data/
│   └── churn_pred.csv          ← Dataset original (10.000 clientes)
├── models/                     ← Gerado automaticamente após train.py
│   ├── pipeline.joblib
│   └── feature_importance.joblib
├── src/
│   ├── train.py                ← Script de treinamento e persistência do modelo
│   └── main.py                 ← Aplicação core (Predição + Integração Gemini)
├── .env                        ← Variáveis de ambiente (Segurança)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Dicionário de Dados (Features)

| Coluna | Descrição Comercial | Impacto no Modelo |
|---|---|---|
| `credit_score` | Pontuação de crédito (350–850) | Scores muito baixos geram atritos com limites, elevando a insatisfação. |
| `country` | França, Alemanha ou Espanha | Fator geográfico/operacional do banco europeu. |
| `gender` | Gênero do cliente | Variável demográfica. |
| `age` | Idade em anos | Clientes com idade mais avançada apresentam maior taxa de churn neste dataset específico. |
| `tenure` | Tempo de relacionamento (0–10 anos) | Clientes muito recentes tendem a ter menor fidelidade ao banco. |
| `balance` | Saldo médio na conta | Saldo baixo ou zerado é um sinal agudo de conta inativa/desengajamento. |
| `products_number` | Quantidade de produtos contratados | Quantos mais produtos (cartão, seguro, consórcio), maior a barreira de saída. |
| `credit_card` | Possui cartão de crédito? (1=Sim, 0=Não) | Facilita a transacionalidade diária. |
| `active_member` | É um cliente ativo? (1=Sim, 0=Não) | A inatividade é o preditor mais forte de que o cliente já abandonou o banco mentalmente. |
| `estimated_salary` | Renda anual estimada | Calibra a oferta que o banco pode fazer na tentativa de retenção. |
| `churn` | **Variável Alvo (Target)** | 1 = Cancelou a conta, 0 = Permaneceu. |

---

## 💡 Destaques de Engenharia de Software

- **Pipeline MLOps com Joblib:** O pré-processamento (`StandardScaler` e `OneHotEncoder`) foi encapsulado em um pipeline scikit-learn. Isso garante que os dados do mundo real em inferência sofram exatamente as mesmas transformações matemáticas do treinamento, eliminando o temido *Data Leakage*.
- **Memória Conversacional (Agentic AI):** A integração com o SDK do Gemini mantém o contexto do chat. O gerente do banco pode dar comandos como *"Achei o tom muito formal, refaça usando uma linguagem mais jovem"* sem precisar reenviar os dados do cliente.
- **Arquitetura Human-in-the-loop:** Sistemas de IA geradores de ações diretas no cliente precisam de curadoria. O design do sistema bloqueia disparos automatizados, exigindo a aprovação do usuário final.

---

## 👨‍💻 Autor

**Felipe Cezar**
* Desenvolvedor de Software | Especialista em Dados
* Conecte-se comigo no [LinkedIn](https://www.linkedin.com/in/felipecezarcruz/)
