"""
train.py — Treinamento do Modelo de Churn

Responsabilidades:
  - Carregar e pré-processar o dataset
  - Treinar um Random Forest Classifier
  - Avaliar o modelo (Acurácia, F1-Score, relatório completo)
  - Salvar o pipeline completo e a importância das variáveis em /models
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---------------------------------------------------------------------------
# Configuração de Caminhos
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "churn_pred.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "pipeline.joblib")
FEATURE_IMPORTANCE_PATH = os.path.join(MODELS_DIR, "feature_importance.joblib")


# ---------------------------------------------------------------------------
# 1. Carregamento e Limpeza dos Dados
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    print(f"[1/5] Carregando dados de: {path}")
    df = pd.read_csv(path)
    print(f"      Shape original: {df.shape}")
    print(f"      Distribuição de churn:\n{df['churn'].value_counts().to_string()}\n")

    # Remover colunas sem valor preditivo
    df = df.drop(columns=["customer_id"])
    return df


# ---------------------------------------------------------------------------
# 2. Definição das Colunas por Tipo
# ---------------------------------------------------------------------------
NUMERICAL_COLS = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "products_number",
    "credit_card",
    "active_member",
    "estimated_salary",
]
CATEGORICAL_COLS = ["country", "gender"]
TARGET_COL = "churn"


# ---------------------------------------------------------------------------
# 3. Construção do Pipeline de Pré-processamento + Modelo
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    """
    Constrói um pipeline sklearn com:
    - StandardScaler para colunas numéricas
    - OneHotEncoder para colunas categóricas
    - RandomForestClassifier como estimador final
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


# ---------------------------------------------------------------------------
# 4. Extração dos Nomes das Features Após Transformação
# ---------------------------------------------------------------------------
def get_feature_names(pipeline: Pipeline) -> list[str]:
    """
    Retorna a lista de nomes das features na ordem em que o
    ColumnTransformer as entrega ao classificador.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    ohe = preprocessor.named_transformers_["cat"]
    ohe_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
    return NUMERICAL_COLS + ohe_feature_names


# ---------------------------------------------------------------------------
# 5. Treinamento e Avaliação
# ---------------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame) -> Pipeline:
    print("[2/5] Preparando dados para treinamento...")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras\n")

    print("[3/5] Treinando Random Forest...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("      Treinamento concluído.\n")

    print("[4/5] Avaliando o modelo no conjunto de teste:")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"      Acurácia : {acc:.4f}")
    print(f"      F1-Score : {f1:.4f}")
    print("\n--- Relatório Completo ---")
    print(classification_report(y_test, y_pred, target_names=["Não Churn", "Churn"]))

    return pipeline


# ---------------------------------------------------------------------------
# 6. Extração e Exibição da Importância das Variáveis
# ---------------------------------------------------------------------------
def extract_feature_importance(pipeline: Pipeline) -> dict:
    """
    Extrai as importâncias de cada feature do Random Forest e retorna
    um dicionário ordenado {feature_name: importance} do maior para o menor.
    """
    feature_names = get_feature_names(pipeline)
    importances = pipeline.named_steps["classifier"].feature_importances_

    importance_dict = dict(zip(feature_names, importances))
    importance_dict = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )

    print("[5/5] Importância das Variáveis (Top 10):")
    for i, (feat, imp) in enumerate(list(importance_dict.items())[:10], 1):
        bar = "█" * int(imp * 50)
        print(f"      {i:>2}. {feat:<25} {imp:.4f} {bar}")
    print()

    return importance_dict


# ---------------------------------------------------------------------------
# 7. Persistência
# ---------------------------------------------------------------------------
def save_artifacts(pipeline: Pipeline, feature_importance: dict) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"Pipeline salvo em:             {PIPELINE_PATH}")

    joblib.dump(feature_importance, FEATURE_IMPORTANCE_PATH)
    print(f"Feature importance salva em:   {FEATURE_IMPORTANCE_PATH}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Motor de Retenção — Treinamento do Modelo de Churn")
    print("=" * 60 + "\n")

    df = load_data(DATA_PATH)
    pipeline = train_and_evaluate(df)
    feature_importance = extract_feature_importance(pipeline)
    save_artifacts(pipeline, feature_importance)

    print("\nTreinamento finalizado com sucesso!")
    print("Execute 'python src/main.py' para iniciar a aplicação.\n")


if __name__ == "__main__":
    main()
