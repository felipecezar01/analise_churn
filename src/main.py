"""
main.py — Motor de Retenção Ativa com IA Agêntica

Responsabilidades:
  - Carregar o pipeline treinado e a importância das variáveis
  - Simular 3 perfis de clientes no terminal
  - Para clientes em Alto Risco: acionar o Gemini para gerar oferta personalizada
  - Gerenciar loop de aprovação humana (Aprovar / Rejeitar / Mudar tom)

CONFIGURAÇÃO DA API KEY:
  1. Crie um arquivo '.env' na raiz do projeto (já existe um modelo).
  2. Preencha: GEMINI_API_KEY=sua_chave_aqui
  3. O arquivo '.env' está no .gitignore e NUNCA deve ser commitado.
  Obtenha sua chave gratuitamente em: https://aistudio.google.com/app/apikey
"""

import os
import sys
import joblib
import pandas as pd
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env para os.environ
load_dotenv()

import google.generativeai as genai

# ---------------------------------------------------------------------------
# Configuração de Caminhos
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "pipeline.joblib")
FEATURE_IMPORTANCE_PATH = os.path.join(BASE_DIR, "models", "feature_importance.joblib")

# ---------------------------------------------------------------------------
# Limiares de Risco
# ---------------------------------------------------------------------------
THRESHOLD_MODERATE = 0.50   # acima de 50% = risco moderado
THRESHOLD_HIGH = 0.70        # acima de 70% = alto risco

# ---------------------------------------------------------------------------
# 3 Perfis de Clientes Fictícios para Simulação
# ---------------------------------------------------------------------------
CUSTOMER_PROFILES = {
    "1": {
        "name": "Carlos",
        "data": {
            "credit_score": 780,
            "country": "France",
            "gender": "Male",
            "age": 35,
            "tenure": 7,
            "balance": 95000.0,
            "products_number": 2,
            "credit_card": 1,
            "active_member": 1,
            "estimated_salary": 85000.0,
        },
    },
    "2": {
        "name": "Beatriz",
        "data": {
            "credit_score": 580,
            "country": "Spain",
            "gender": "Female",
            "age": 47,
            "tenure": 3,
            "balance": 30000.0,
            "products_number": 1,
            "credit_card": 1,
            "active_member": 0,
            "estimated_salary": 62000.0,
        },
    },
    "3": {
        "name": "Ana",
        "data": {
            "credit_score": 420,
            "country": "Germany",
            "gender": "Female",
            "age": 58,
            "tenure": 1,
            "balance": 0.0,
            "products_number": 1,
            "credit_card": 0,
            "active_member": 0,
            "estimated_salary": 41000.0,
        },
    },
}

# ---------------------------------------------------------------------------
# Mapeamento de Variáveis Críticas → Estratégias de Retenção
# ---------------------------------------------------------------------------
RETENTION_STRATEGIES = {
    "active_member": {
        "business_name": "Inatividade do cliente",
        "strategy": "programa de cashback progressivo para reativar o uso dos serviços bancários",
    },
    "balance": {
        "business_name": "Saldo zerado na conta",
        "strategy": "isenção de tarifas por 6 meses e taxa especial de CDB para incentivar novos depósitos",
    },
    "products_number": {
        "business_name": "Baixa adesão a produtos",
        "strategy": "isenção de anuidade no cartão de crédito premium e acesso gratuito a conta investimento",
    },
    "age": {
        "business_name": "Perfil de idade avançada com menor engajamento",
        "strategy": "atendimento exclusivo via gerente dedicado e plano de previdência sem taxa de carregamento",
    },
    "credit_score": {
        "business_name": "Score de crédito baixo",
        "strategy": "programa de renegociação de dívidas com juros reduzidos e consultoria financeira gratuita",
    },
}

# Descrições amigáveis das variáveis para exibição ao usuário
FEATURE_DESCRIPTIONS = {
    "age":            "Idade do cliente — clientes mais velhos tendem a migrar para bancos tradicionais",
    "active_member":  "Engajamento — clientes inativos param de usar o banco antes de sair",
    "balance":        "Saldo em conta — saldo zerado indica desvinculação financeira do banco",
    "products_number":"Nº de produtos — quem tem só 1 produto tem menos razões para ficar",
    "credit_score":   "Score de crédito — score baixo gera frustração com limites e negativas",
    "tenure":         "Tempo de relacionamento — clientes novos ou muito antigos são mais voláteis",
    "estimated_salary":"Renda estimada — influencia o quanto o cliente valoriza benefícios financeiros",
    "credit_card":    "Posse de cartão — sem cartão, o vínculo diário com o banco é menor",
}

# Valores que indicam risco para cada variável
RISKY_VALUES = {
    "active_member": lambda v: v == 0,
    "balance": lambda v: v == 0.0,
    "products_number": lambda v: v == 1 or v >= 4,
    "age": lambda v: v > 50,
    "credit_score": lambda v: v < 550,
}


# ---------------------------------------------------------------------------
# Carregamento dos Artefatos
# ---------------------------------------------------------------------------
def load_artifacts():
    if not os.path.exists(PIPELINE_PATH):
        print("ERRO: Modelo não encontrado. Execute 'python src/train.py' primeiro.")
        sys.exit(1)

    pipeline = joblib.load(PIPELINE_PATH)
    feature_importance = joblib.load(FEATURE_IMPORTANCE_PATH)
    return pipeline, feature_importance


# ---------------------------------------------------------------------------
# Predição
# ---------------------------------------------------------------------------
def predict(pipeline, customer_data: dict) -> tuple[float, int]:
    """
    Retorna (probabilidade_de_churn, classe_predita).
    """
    df = pd.DataFrame([customer_data])
    proba = pipeline.predict_proba(df)[0][1]   # probabilidade da classe 1 (churn)
    label = int(proba >= THRESHOLD_MODERATE)
    return proba, label


# ---------------------------------------------------------------------------
# Identificação da Variável Crítica
# ---------------------------------------------------------------------------
def identify_critical_variable(
    customer_data: dict, feature_importance: dict
) -> tuple[str, str, str]:
    """
    Percorre as features em ordem de importância e retorna a primeira
    cujo valor no perfil do cliente é considerado 'de risco'.

    Retorna: (feature_key, business_name, strategy)
    """
    for feature in feature_importance:
        if feature in RISKY_VALUES and feature in customer_data:
            value = customer_data[feature]
            if RISKY_VALUES[feature](value):
                if feature in RETENTION_STRATEGIES:
                    info = RETENTION_STRATEGIES[feature]
                    return feature, info["business_name"], info["strategy"]

    # Fallback: retorna a feature mais importante independentemente do valor
    top_feature = list(feature_importance.keys())[0]
    info = RETENTION_STRATEGIES.get(
        top_feature,
        {
            "business_name": top_feature,
            "strategy": "benefícios exclusivos de fidelidade",
        },
    )
    return top_feature, info["business_name"], info["strategy"]


# ---------------------------------------------------------------------------
# Integração com Gemini
# ---------------------------------------------------------------------------
def setup_gemini() -> genai.GenerativeModel:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "SUA_CHAVE_AQUI":
        print("\nERRO: Chave de API do Gemini não configurada.")
        print("Edite src/main.py e insira sua chave em os.environ['GEMINI_API_KEY'].")
        sys.exit(1)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-3-flash-preview")


def build_retention_prompt(
    customer_data: dict,
    churn_proba: float,
    critical_var_name: str,
    business_name: str,
    strategy: str,
) -> str:
    """
    Monta o prompt inicial para o Gemini agir como gerente de retenção.
    """
    return f"""Você é um gerente de relacionamento sênior de um banco digital, especialista em retenção de clientes.
Você precisa redigir um e-mail de retenção personalizado e empático para um cliente em risco.

## Dados do Cliente

- País: {customer_data['country']}
- Idade: {customer_data['age']} anos
- Score de Crédito: {customer_data['credit_score']}
- Saldo na conta: R$ {customer_data['balance']:,.2f}
- Número de produtos: {customer_data['products_number']}
- Membro ativo: {'Sim' if customer_data['active_member'] == 1 else 'Não'}
- Salário estimado: R$ {customer_data['estimated_salary']:,.2f}

## Diagnóstico do Sistema de IA

- Probabilidade de abandono (Churn): **{churn_proba:.1%}**
- Fator crítico identificado: **{business_name}** (variável: `{critical_var_name}`)

## Instrução

Redija um e-mail de retenção **curto (máximo 150 palavras)**, **cordial** e **personalizado**.
O e-mail DEVE:
1. Reconhecer o valor do cliente para o banco (sem mencioná-lo como "cliente em risco")
2. Oferecer especificamente o seguinte benefício como solução para o problema identificado:
   **{strategy}**
3. Incluir uma call-to-action clara (ex: "Responda este e-mail" ou "Acesse o app")
4. Assinar como "Equipe de Relacionamento Premium"

Escreva APENAS o corpo do e-mail, sem assunto e sem explicações adicionais."""


# ---------------------------------------------------------------------------
# Loop de Aprovação Humana
# ---------------------------------------------------------------------------
def approval_loop(chat: genai.ChatSession, initial_prompt: str) -> None:
    """
    Envia o prompt inicial, exibe a resposta e entra no loop
    Aprovar / Rejeitar / Mudar tom.
    """
    print("\n" + "=" * 60)
    print("  Gerando oferta de retenção via Gemini...")
    print("=" * 60)

    response = chat.send_message(initial_prompt)
    email_text = response.text

    while True:
        print("\n--- RASCUNHO DO E-MAIL DE RETENÇÃO ---\n")
        print(email_text)
        print("\n" + "-" * 40)
        print("[A] Aprovar e enviar")
        print("[R] Rejeitar e cancelar")
        print("[M] Mudar o tom ou ajustar o conteúdo")
        choice = input("\nSua decisão: ").strip().upper()

        if choice == "A":
            print("\n✓ E-mail enviado com sucesso para o cliente!")
            print("  Ação de retenção registrada no CRM.")
            break

        elif choice == "R":
            print("\n✗ Envio cancelado. Nenhuma ação foi tomada.")
            break

        elif choice == "M":
            feedback = input(
                "\nDescreva o que deseja mudar (ex: 'mais formal', 'incluir prazo', 'tom mais urgente'): "
            ).strip()
            if not feedback:
                print("Nenhuma instrução fornecida. Tente novamente.")
                continue

            refinement_prompt = (
                f"Reescreva o e-mail acima incorporando o seguinte ajuste solicitado pelo gerente: "
                f'"{feedback}". '
                f"Mantenha o mesmo benefício e o mesmo tamanho máximo de 150 palavras."
            )
            print("\nGerando nova versão...")
            response = chat.send_message(refinement_prompt)
            email_text = response.text

        else:
            print("Opção inválida. Digite A, R ou M.")


# ---------------------------------------------------------------------------
# Interface CLI
# ---------------------------------------------------------------------------
def print_intro(feature_importance: dict) -> None:
    """Exibe a introdução narrativa do sistema antes do menu principal."""
    top3 = list(feature_importance.keys())[:3]

    print("\n" + "=" * 60)
    print("  Motor de Retenção Ativa com IA Agêntica")
    print("=" * 60)
    print("""
  Bem-vindo ao sistema de retenção preditiva.

  Este modelo foi treinado com dados históricos de 10.000 clientes,
  aprendendo padrões de comportamento — quem permaneceu fiel ao banco
  e quem acabou cancelando a conta.

  Após analisar todas as variáveis disponíveis, o algoritmo descobriu
  que as 3 que mais causam evasão (churn) são:\n""")

    for i, feature in enumerate(top3, 1):
        desc = FEATURE_DESCRIPTIONS.get(feature, feature)
        print(f"    {i}. {desc}")

    print(
        "\n"
        "  Imagine que você é o Diretor de Relacionamento deste banco.\n"
        "  Você tem 3 clientes no seu radar hoje: Ana, Beatriz e Carlos.\n"
        "  Vamos usar a IA para analisar o perfil de cada um e, quando\n"
        "  necessário, criar intervenções de retenção personalizadas.\n"
    )
    input("  Pressione ENTER para acessar o painel de clientes...")


def display_menu() -> str:
    print("\n" + "=" * 60)
    print("  Painel de Clientes — Selecione para Analisar")
    print("=" * 60)
    print("\n  Qual cliente você deseja analisar agora?\n")
    for key, profile in CUSTOMER_PROFILES.items():
        print(f"  [{key}] {profile['name']}")
    print("  [0] Encerrar sessão")
    return input("\nOpção: ").strip()


def display_prediction_result(
    customer_name: str, customer_data: dict, churn_proba: float
) -> None:
    print(f"\n{'─' * 60}")
    print(f"  Analisando perfil de: {customer_name}")
    print(f"{'─' * 60}")

    # Dados legíveis do cliente
    active = "Sim" if customer_data["active_member"] == 1 else "Não"
    card = "Sim" if customer_data["credit_card"] == 1 else "Não"
    print(
        f"\n  Idade          : {customer_data['age']} anos"
        f"\n  País           : {customer_data['country']}"
        f"\n  Score Crédito  : {customer_data['credit_score']}"
        f"\n  Saldo          : R$ {customer_data['balance']:>12,.2f}"
        f"\n  Produtos       : {customer_data['products_number']}"
        f"\n  Cartão crédito : {card}"
        f"\n  Membro ativo   : {active}"
        f"\n  Salário est.   : R$ {customer_data['estimated_salary']:>12,.2f}"
    )

    print(f"\n  {'─' * 40}")
    print("  Iniciando análise preditiva do perfil...")
    print(f"  {'─' * 40}\n")

    print(f"  Probabilidade de Churn : {churn_proba:.1%}")

    if churn_proba < THRESHOLD_MODERATE:
        status = "SEGURO"
        hint = "(baixo risco)"
    elif churn_proba < THRESHOLD_HIGH:
        status = "RISCO MODERADO"
        hint = "(monitorar)"
    else:
        status = "ALTO RISCO"
        hint = "(ação imediata necessária)"

    print(f"  Status                 : {status} {hint}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    pipeline, feature_importance = load_artifacts()
    gemini_model = None  # inicializar Gemini apenas se necessário

    print_intro(feature_importance)

    while True:
        choice = display_menu()

        if choice == "0":
            print("\nEncerrando o Motor de Retenção. Até logo!\n")
            break

        if choice not in CUSTOMER_PROFILES:
            print("Opção inválida. Tente novamente.")
            continue

        profile = CUSTOMER_PROFILES[choice]
        customer_name = profile["name"]
        customer_data = profile["data"]

        # --- Predição ---
        churn_proba, _ = predict(pipeline, customer_data)
        display_prediction_result(customer_name, customer_data, churn_proba)

        # --- Decisão baseada no limiar ---
        if churn_proba < THRESHOLD_MODERATE:
            print("\n  Perfil saudável. Nenhuma intervenção necessária no momento.")

        elif churn_proba < THRESHOLD_HIGH:
            print("\n  Recomendação: Agendar revisão de portfólio nos próximos 30 dias.")
            print("  Sugestão: Oferecer upgrade de conta sem custo adicional.")

        else:
            # --- Fluxo de Alto Risco: Acionar Gemini ---
            print(f"\n  ALERTA: {customer_name} está em alto risco de abandono ({churn_proba:.1%})!")

            critical_var, business_name, strategy = identify_critical_variable(
                customer_data, feature_importance
            )
            print(f"\n  Fator crítico identificado : {business_name}")
            print(f"  Estratégia sugerida        : {strategy}")

            # Inicializar Gemini na primeira vez necessária
            if gemini_model is None:
                gemini_model = setup_gemini()

            # Iniciar sessão de chat (mantém histórico para refinamentos)
            chat = gemini_model.start_chat(history=[])

            # Montar prompt e iniciar loop de aprovação
            prompt = build_retention_prompt(
                customer_data=customer_data,
                churn_proba=churn_proba,
                critical_var_name=critical_var,
                business_name=business_name,
                strategy=strategy,
            )
            approval_loop(chat, prompt)

        input("\nPressione ENTER para voltar ao menu...")


if __name__ == "__main__":
    main()
