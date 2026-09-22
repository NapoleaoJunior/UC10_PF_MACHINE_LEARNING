# ============================================================
# PROJETO INTEGRADOR - STUDENT PERFORMANCE PREDICTION
# Problema de Classificação
# ============================================================

# ============================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE


# ============================================================
# 2. CARREGAMENTO DO DATASET
# ============================================================

# Ajuste o caminho caso seu arquivo esteja em outra pasta
df = pd.read_csv("../data/raw/student_perfomance_dataset.csv")

print("\n========== PRIMEIRAS LINHAS ==========")
print(df.head())

print("\n========== DIMENSÕES ==========")
print(df.shape)

print("\n========== INFORMAÇÕES ==========")
print(df.info())



# ============================================================
# 5. ANÁLISE DA VARIÁVEL FINAL_GRADE
# ============================================================

print("\n========== DISTRIBUIÇÃO NOTA_FINAL ==========")
print(df["final_grade"].value_counts())


# Visualização
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="final_grade"
)

plt.title("Distribuição das Notas Finais")
plt.xlabel("Nota Final")
plt.ylabel("Quantidade de Alunos")

plt.tight_layout()
plt.show()


# ============================================================
# 6. CRIAÇÃO DO PROBLEMA DE CLASSIFICAÇÃO
# ============================================================

# Neste projeto:
#
# 0 = Baixo desempenho
# 1 = Bom desempenho
#
# Consideraremos A e B como bom desempenho.
# C, D e F serão considerados baixo desempenho.

df["target"] = df["final_grade"].apply(
    lambda x: 1 if x in ["A", "B"] else 0
)

print("\n========== DISTRIBUIÇÃO DA CLASSIFICAÇÃO ==========")

print(df["target"].value_counts())

print("\nPercentual:")
print(
    df["target"]
    .value_counts(normalize=True)
    .mul(100)
)


# ============================================================
# 7. VISUALIZAÇÃO DA CLASSIFICAÇÃO
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="target"
)

plt.title("Distribuição das Classes")

plt.xlabel(
    "Classificação "
    "(0 = Baixo desempenho | 1 = Bom desempenho)"
)

plt.ylabel("Quantidade")

plt.tight_layout()
plt.show()


# ============================================================
# 8. DEFINIÇÃO DAS VARIÁVEIS
# ============================================================

# IMPORTANTE:
#
# Não utilizaremos final_grade como entrada,
# pois ela é justamente a informação usada
# para criar o nosso target.
#
# Também não utilizaremos final_exam_score,
# pois isso poderia causar vazamento de informação:
# a nota da prova final já está relacionada
# diretamente ao resultado que queremos prever.

features = [
    "gender",
    "study_time_hours",
    "attendance_percent",
    "sleep_hours",
    "parental_education",
    "internet_access",
    "extracurricular_activities",
    "part_time_job",
    "previous_grade"
]

X = df[features]

y = df["target"]


print("\n========== VARIÁVEIS DE ENTRADA ==========")
print(X.columns.tolist())

print("\n========== VARIÁVEL TARGET ==========")
print(y.name)


# ============================================================
# 9. IDENTIFICAR VARIÁVEIS CATEGÓRICAS E NUMÉRICAS
# ============================================================

categorical_features = [
    "gender",
    "parental_education",
    "internet_access",
    "extracurricular_activities",
    "part_time_job"
]

numeric_features = [
    "study_time_hours",
    "attendance_percent",
    "sleep_hours",
    "previous_grade"
]


# ============================================================
# 10. DIVISÃO TREINO / TESTE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DIVISÃO DOS DADOS ==========")

print("Treinamento:", X_train.shape)
print("Teste:", X_test.shape)


# ============================================================
# 11. PRÉ-PROCESSAMENTO
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# Aplicar o pré-processamento
X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


print("\n========== PRÉ-PROCESSAMENTO ==========")

print(
    "Formato treinamento:",
    X_train_processed.shape
)

print(
    "Formato teste:",
    X_test_processed.shape
)


# ============================================================
# 12. BALANCEAMENTO COM SMOTE
# ============================================================

print("\n========== ANTES DO BALANCEAMENTO ==========")

print(y_train.value_counts())


smote = SMOTE(
    random_state=42
)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_processed,
    y_train
)


print("\n========== DEPOIS DO BALANCEAMENTO ==========")

print(y_train_balanced.value_counts())


# ============================================================
# 13. MODELO 1 - KNN
# ============================================================

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(
    X_train_balanced,
    y_train_balanced
)

y_pred_knn = knn.predict(
    X_test_processed
)


# ============================================================
# 14. MODELO 2 - DECISION TREE
# ============================================================

decision_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

decision_tree.fit(
    X_train_balanced,
    y_train_balanced
)

y_pred_tree = decision_tree.predict(
    X_test_processed
)


# ============================================================
# 15. MODELO 3 - RANDOM FOREST
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(
    X_train_balanced,
    y_train_balanced
)

y_pred_rf = random_forest.predict(
    X_test_processed
)


# ============================================================
# 16. FUNÇÃO PARA AVALIAÇÃO
# ============================================================

def avaliar_modelo(nome, y_real, y_pred):

    accuracy = accuracy_score(
        y_real,
        y_pred
    )

    precision = precision_score(
        y_real,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_real,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_real,
        y_pred,
        zero_division=0
    )

    return {
        "Modelo": nome,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }


# ============================================================
# 17. COMPARAÇÃO DOS MODELOS
# ============================================================

resultados = []

resultados.append(
    avaliar_modelo(
        "KNN",
        y_test,
        y_pred_knn
    )
)

resultados.append(
    avaliar_modelo(
        "Decision Tree",
        y_test,
        y_pred_tree
    )
)

resultados.append(
    avaliar_modelo(
        "Random Forest",
        y_test,
        y_pred_rf
    )
)


resultados_df = pd.DataFrame(
    resultados
)


print("\n========== RESULTADOS ==========")

print(
    resultados_df.to_string(
        index=False
    )
)


# ============================================================
# 18. GRÁFICO DE COMPARAÇÃO
# ============================================================

resultados_melted = resultados_df.melt(
    id_vars="Modelo",
    var_name="Métrica",
    value_name="Valor"
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=resultados_melted,
    x="Modelo",
    y="Valor",
    hue="Métrica"
)

plt.title("Comparação dos Modelos")

plt.ylabel("Score")

plt.ylim(0, 1)

plt.tight_layout()
plt.show()


# ============================================================
# 19. IDENTIFICAR O MELHOR MODELO
# ============================================================

melhor_modelo = resultados_df.loc[
    resultados_df["F1-Score"].idxmax()
]

print("\n========== MELHOR MODELO ==========")

print(
    melhor_modelo
)


# ============================================================
# 20. MATRIZ DE CONFUSÃO - KNN
# ============================================================

cm_knn = confusion_matrix(
    y_test,
    y_pred_knn
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_knn,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Matriz de Confusão - KNN")

plt.xlabel("Previsto")
plt.ylabel("Real")

plt.tight_layout()
plt.show()


# ============================================================
# 21. MATRIZ DE CONFUSÃO - DECISION TREE
# ============================================================

cm_tree = confusion_matrix(
    y_test,
    y_pred_tree
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_tree,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title("Matriz de Confusão - Decision Tree")

plt.xlabel("Previsto")
plt.ylabel("Real")

plt.tight_layout()
plt.show()


# ============================================================
# 22. MATRIZ DE CONFUSÃO - RANDOM FOREST
# ============================================================

cm_rf = confusion_matrix(
    y_test,
    y_pred_rf
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Oranges"
)

plt.title("Matriz de Confusão - Random Forest")

plt.xlabel("Previsto")
plt.ylabel("Real")

plt.tight_layout()
plt.show()


# ============================================================
# 23. RELATÓRIO DO MELHOR MODELO
# ============================================================

# Neste exemplo usamos Random Forest.
# Caso outro modelo tenha apresentado melhor F1,
# altere y_pred_rf pelo modelo escolhido.

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred_rf,
        target_names=[
            "Baixo desempenho",
            "Bom desempenho"
        ],
        zero_division=0
    )
)
