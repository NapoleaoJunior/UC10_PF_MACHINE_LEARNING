"""
PI - UC 10 | Aprendizado de Máquina
Classificação binária: Aprovado (1) / Reprovado (0)

Pipeline (segue as orientações do professor):
  Aquisição -> Limpeza/Tratamento -> Balanceamento (SMOTE, só no treino)
  -> Split 70/30 estratificado -> StandardScaler -> PCA (2,3,4,5)
  -> KNN (k = 3, 5, 7) + Random Forest -> Avaliação (Acc/Prec/Rec/F1)
  -> Matriz de Confusão -> Melhor modelo (F1-macro) -> Previsão
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# SMOTE é opcional (se não tiver, o código segue com class_weight)
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_OK = True
except ImportError:
    IMBLEARN_OK = False

# ------------------------------------------------------------------
# Configurações gerais
# ------------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
RANDOM_STATE = 42

CSV_PATH = r"desempenho_estudantil\tratamento_dados_binario.csv"
OUT_DIR  = "reports"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1) AQUISIÇÃO DOS DADOS
# ------------------------------------------------------------------
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
print(f"[1] Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
print(df.head(), "\n")

# ------------------------------------------------------------------
# 2) LIMPEZA E TRATAMENTO
# ------------------------------------------------------------------
# 2.1 Remover duplicados
antes = len(df)
df = df.drop_duplicates()
print(f"[2.1] Duplicados removidos: {antes - len(df)}")

# 2.2 Remover linhas com dados omissos (professor: deletar a amostra)
antes = len(df)
df = df.dropna()
print(f"[2.2] Linhas com dados omissos removidas: {antes - len(df)}")

# 2.3 Garantir que o target é binário (0/1)
target_col = "nota_final"
valores_unicos = sorted(df[target_col].dropna().unique())
print(f"[2.3] Valores únicos em '{target_col}': {valores_unicos}")

# Caso A: já é 0/1
if set(valores_unicos) <= {0, 1}:
    print("[2.3] Target já está binário (0/1). Mantendo.")

# Caso B: são letras de nota (A-F)
elif all(isinstance(v, str) for v in valores_unicos):
    mapa_nota = {"A": 1, "B": 1, "C": 1, "D": 0, "F": 0}
    df[target_col] = df[target_col].map(mapa_nota)
    print("[2.3] Target binário convertido de letras (A,B,C=1 | D,F=0).")

# Caso C: notas numéricas 0–100
elif df[target_col].max() > 1:
    df[target_col] = (df[target_col] >= 60).astype(int)
    print("[2.3] Target binarizado por nota de corte (>= 60).")

# Caso D: notas ordinais 0–4
else:
    mediana = df[target_col].median()
    df[target_col] = (df[target_col] >= mediana).astype(int)
    print(f"[2.3] Target binarizado por mediana ({mediana}).")

# Remove qualquer linha que virou NaN no mapeamento (letras desconhecidas)
antes = len(df)
df = df.dropna(subset=[target_col])
if len(df) < antes:
    print(f"[2.3] {antes - len(df)} linhas removidas por labels desconhecidos.")

print(f"[2.3] Distribuição final: {df[target_col].value_counts().to_dict()}\n")

# ------------------------------------------------------------------
# 3) FEATURES E TARGET
# ------------------------------------------------------------------
features = [
    "genero",
    "horas_estudo",
    "percentual_frequencia",
    "horas_sono",
    "escolaridade_pais",
    "acesso_internet",
    "atividades_extracurriculares",
    "trabalho_meio_periodo",
    "nota_anterior",
]
X = df[features].copy()
y = df[target_col].copy()

print("\nDistribuição original das classes:")
print(y.value_counts(), "\n")

# Plot: distribuição das classes (antes)
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=y, palette="Set2", ax=ax)
ax.set_title("Distribuição das classes (antes do balanceamento)")
ax.set_xlabel("Classe (0 = Reprovado | 1 = Aprovado)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "dist_classes_antes.png"), dpi=120)
plt.close()

# ------------------------------------------------------------------
# 4) SPLIT TREINO/TESTE (70/30, estratificado)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
print(f"[4] Treino: {X_train.shape} | Teste: {X_test.shape}")

# ------------------------------------------------------------------
# 5) BALANCEAMENTO (apenas no treino)
# ------------------------------------------------------------------
if IMBLEARN_OK:
    try:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("[5] SMOTE aplicado APENAS no treino.")
    except ValueError as e:
        print(f"[5] SMOTE falhou ({e}) — seguindo sem balanceamento.")
else:
    print("[5] imblearn não instalado — pulando SMOTE "
          "(instale com: pip install imbalanced-learn)")

print("Distribuição pós-balanceamento (treino):")
print(pd.Series(y_train).value_counts(), "\n")

# Plot: distribuição pós-balanceamento
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=pd.Series(y_train), palette="Set2", ax=ax)
ax.set_title("Distribuição das classes (pós-balanceamento)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "dist_classes_depois.png"), dpi=120)
plt.close()

# ------------------------------------------------------------------
# 6) PADRONIZAÇÃO (Scaler ajustado SÓ no treino)
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print("[6] StandardScaler ajustado no treino e aplicado no teste.\n")

# ------------------------------------------------------------------
# 7) PCA + KNN + RANDOM FOREST
# ------------------------------------------------------------------
componentes = [2, 3, 4, 5]
knn_ks      = [3, 5, 7]

# 7.1 Visualização PCA 2D (com as classes)
pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_pca2 = pca2.fit_transform(X_train_s)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_train_pca2[:, 0], y=X_train_pca2[:, 1],
                hue=pd.Series(y_train).values, palette="Set1", alpha=0.7)
plt.title("PCA 2D — projeção das classes (treino)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_2d.png"), dpi=120)
plt.close()

# 7.2 Variância explicada por componente
pca_full = PCA(random_state=RANDOM_STATE).fit(X_train_s)
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_, alpha=0.7, label="Individual")
plt.step(range(1, len(pca_full.explained_variance_ratio_) + 1),
         np.cumsum(pca_full.explained_variance_ratio_),
         where="mid", color="red", label="Acumulada")
plt.title("Variância explicada por componente principal")
plt.xlabel("Componente"); plt.ylabel("Variância explicada")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_variancia.png"), dpi=120)
plt.close()

# 7.3 Combinar PCA(n) x KNN(k)
resultados = []

for n in componentes:
    pca = PCA(n_components=n, random_state=RANDOM_STATE)
    X_tr = pca.fit_transform(X_train_s)
    X_te = pca.transform(X_test_s)
    var  = pca.explained_variance_ratio_.sum()

    for k in knn_ks:
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_tr, y_train)
        y_pred = modelo.predict(X_te)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_test, y_pred,    average="macro", zero_division=0)
        f1   = f1_score(y_test, y_pred,        average="macro", zero_division=0)

        resultados.append({
            "pca": n, "var_explicada": round(var, 4), "k": k,
            "acuracia": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1_macro": round(f1, 4)
        })

        # Matriz de confusão (seaborn heatmap)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Reprovado", "Aprovado"],
                    yticklabels=["Reprovado", "Aprovado"])
        plt.title(f"Matriz de Confusão — PCA {n} | KNN k={k}\nAcc={acc:.2%}")
        plt.xlabel("Previsto"); plt.ylabel("Real")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"cm_pca{n}_k{k}.png"), dpi=120)
        plt.close()

print("[7.3] Resultados parciais (KNN):")
print(pd.DataFrame(resultados).sort_values("f1_macro", ascending=False)
      .to_string(index=False), "\n")

# ------------------------------------------------------------------
# 7.4) RANDOM FOREST (comparação com KNN — exigido pelo professor)
# ------------------------------------------------------------------
for n in componentes:
    pca = PCA(n_components=n, random_state=RANDOM_STATE)
    X_tr = pca.fit_transform(X_train_s)
    X_te = pca.transform(X_test_s)

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    rf.fit(X_tr, y_train)
    y_pred_rf = rf.predict(X_te)

    acc  = accuracy_score(y_test, y_pred_rf)
    prec = precision_score(y_test, y_pred_rf, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred_rf,    average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred_rf,        average="macro", zero_division=0)

    resultados.append({
        "pca": n,
        "var_explicada": round(pca.explained_variance_ratio_.sum(), 4),
        "k": "RF",
        "acuracia": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_macro": round(f1, 4)
    })

    # Matriz de confusão do RF
    cm = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Reprovado", "Aprovado"],
                yticklabels=["Reprovado", "Aprovado"])
    plt.title(f"Matriz de Confusão — PCA {n} | Random Forest\nAcc={acc:.2%}")
    plt.xlabel("Previsto"); plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"cm_pca{n}_rf.png"), dpi=120)
    plt.close()

# Consolidar resultados finais (KNN + RF)
df_res = (pd.DataFrame(resultados)
          .sort_values("f1_macro", ascending=False)
          .reset_index(drop=True))

print("[7.4] Resultados finais (KNN + Random Forest):")
print(df_res.to_string(index=False), "\n")
df_res.to_csv(os.path.join(OUT_DIR, "comparacao_modelos.csv"),
              index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------
# 8) GRÁFICOS COMPARATIVOS
# ------------------------------------------------------------------
# Gráfico 1: só KNN
df_knn = df_res[df_res["k"] != "RF"].copy()

if not df_knn.empty:
    df_knn["k"] = df_knn["k"].astype(int)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_knn, x="k", y="f1_macro", hue="pca", palette="viridis")
    plt.axhline(0.5, color="red", linestyle="--", label="Baseline 50%")
    plt.title("KNN — F1-macro por k × PCA")
    plt.ylim(0, 1)
    plt.legend(title="Componentes PCA")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comparacao_knn.png"), dpi=120)
    plt.close()
else:
    print("[8] Sem resultados de KNN — pulando gráfico do KNN.")

# Gráfico 2: KNN × Random Forest (por PCA)
df_res_plot = df_res.copy()
df_res_plot["k"] = df_res_plot["k"].astype(str)
ordem_k = ["3", "5", "7", "RF"]

plt.figure(figsize=(10, 6))
sns.barplot(data=df_res_plot, x="pca", y="f1_macro", hue="k",
            hue_order=ordem_k, palette="Set2")
plt.axhline(0.5, color="red", linestyle="--", label="Baseline 50%")
plt.title("F1-macro — KNN × Random Forest por nº de componentes PCA")
plt.ylim(0, 1)
plt.legend(title="Classificador (k ou RF)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "comparacao_knn_vs_rf.png"), dpi=120)
plt.close()

# ------------------------------------------------------------------
# 9) MELHOR MODELO (por F1-macro)
# ------------------------------------------------------------------
melhor = df_res.iloc[0]
print("[9] Melhor configuração encontrada:")
print(melhor.to_dict())

if melhor["acuracia"] < 0.5:
    print("⚠️  Acurácia < 50% — considere outro classificador/parâmetros.\n")

# --- Re-treina o melhor modelo respeitando o classificador vencedor ---
pca_best = PCA(n_components=int(melhor["pca"]), random_state=RANDOM_STATE)
X_tr_best = pca_best.fit_transform(X_train_s)
X_te_best = pca_best.transform(X_test_s)

if melhor["k"] == "RF":
    print("[9] Classificador vencedor: Random Forest")
    modelo_best = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
else:
    print(f"[9] Classificador vencedor: KNN (k={int(melhor['k'])})")
    modelo_best = KNeighborsClassifier(n_neighbors=int(melhor["k"]))

modelo_best.fit(X_tr_best, y_train)
y_pred_best = modelo_best.predict(X_te_best)

print("\nRelatório final do melhor modelo:")
print(classification_report(
    y_test, y_pred_best,
    labels=[0, 1],
    target_names=["Reprovado", "Aprovado"],
    zero_division=0
))

# Matriz de confusão final do melhor modelo
cm_best = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_best, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["Reprovado", "Aprovado"],
            yticklabels=["Reprovado", "Aprovado"])
plt.title(f"Matriz de Confusão — MELHOR MODELO\n"
          f"PCA {int(melhor['pca'])} | "
          f"{'RF' if melhor['k'] == 'RF' else f'KNN k={int(melhor[chr(39)+chr(107)+chr(39)])}' if False else melhor['k']}\n"
          f"Acc={melhor['acuracia']:.2%} | F1-macro={melhor['f1_macro']:.4f}")
plt.xlabel("Previsto"); plt.ylabel("Real")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cm_melhor_modelo.png"), dpi=120)
plt.close()

# ------------------------------------------------------------------
# 10) EXEMPLO — previsão para um novo aluno
# ------------------------------------------------------------------
novo_aluno = pd.DataFrame({
    "genero": [1],
    "horas_estudo": [4.5],
    "percentual_frequencia": [90],
    "horas_sono": [7],
    "escolaridade_pais": [2],
    "acesso_internet": [1],
    "atividades_extracurriculares": [1],
    "trabalho_meio_periodo": [0],
    "nota_anterior": [75],
})
novo_s   = scaler.transform(novo_aluno)
novo_pca = pca_best.transform(novo_s)
pred     = modelo_best.predict(novo_pca)[0]
print(f"\n[10] Previsão para o novo aluno: "
      f"{'APROVADO ✅' if pred == 1 else 'REPROVADO ❌'}")