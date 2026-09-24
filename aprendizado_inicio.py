from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
IMBLEARN_OK, RANDOM_STATE = True, 42
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_OK = True
except ImportError:
    IMBLEARN_OK = False
# Carregar CSV traduzido
df = pd.read_csv(r"desempenho_estudantil\tratamento_dados_binario_inicio.csv")
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
    "pontuacao_prova_final"
]
X = df[features]
y = df["nota_final"]
print("[6] StandardScaler ajustado no treino e aplicado no teste.\n")
# seperar os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
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
# normaliza o espaço de caracteristicas
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
# criar o modelo de Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
print(df["nota_final"].unique())


# treinar com 70%
modelo.fit(X_train, y_train)

# testar com 30%
y_pred = modelo.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)
print("\nAcurácia:")
print(f"{acuracia:.2%}")

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("Valores reais:")
print(y_test.head())

print("\nValores previstos:")
print(y_pred[:5])
# Novo aluno
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
    "pontuacao_prova_final": [80]
})
previsao = modelo.predict(novo_aluno)
print("Previsão:", previsao[0])
if previsao[0] == 1:
    print("Aluno APROVADO ✅")
else:
    print("Aluno REPROVADO ❌")
print("Nota prevista:", previsao[0])
print("nota_final:", modelo.predict_proba(novo_aluno))
# Fazer previsões utilizando apenas os 30% de teste
y_pred = modelo.predict(X_test)
# Mostrar algumas previsões
acuracia = accuracy_score(y_test, y_pred)
print("\n========================")
print("RESULTADO FINAL")
print("========================")
print(f"Acurácia do Modelo: {acuracia:.2%}")
if acuracia >= 0.90:
    print("Modelo EXCELENTE ⭐")
elif acuracia >= 0.80:
    print("Modelo BOM ✅")
elif acuracia >= 0.70:
    print("Modelo ACEITÁVEL ⚠️")
else:
    print("Modelo precisa melhorar ❌")
