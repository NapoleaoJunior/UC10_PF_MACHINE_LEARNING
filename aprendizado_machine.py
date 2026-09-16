import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Carregar CSV traduzido
df = pd.read_csv(r"desempenho_estudantil\tratamento_dados_binario.csv")
print(df["nota_final"].value_counts())
X = df[
    [
        "genero",
        "horas_estudo",
        "percentual_frequencia",
        "horas_sono",
        "escolaridade_pais",
        "acesso_internet",
        "atividades_extracurriculares",
        "trabalho_meio_periodo",
        "nota_anterior"
    ]
]
# variavel alvo
y = df["nota_final"]
# normaliza o espaço de caracteristicas
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)
# separar os dados em treino e teste mantendo a proporção das classes
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
# criar o modelo de Random Forest
modelo = RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced")
print(df["nota_final"].unique())
# criar o modelo de Random Forest

# treinar com 70%
modelo.fit(X_train, y_train)

# testar com 30%
y_pred = modelo.predict(X_test)
# Avaliação do modelo
acuracia = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acuracia:.2%}")
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("Valores reais:")
print(y_test.head())
print("Score do modelo:", modelo.score(X_test, y_test))
print("\nValores previstos:")
print(y_pred[:5])
novo_aluno = pd.DataFrame({
    "genero": [1],
    "horas_estudo": [4.5],
    "percentual_frequencia": [90],
    "horas_sono": [7],
    "escolaridade_pais": [2],
    "acesso_internet": [1],
    "atividades_extracurriculares": [1],
    "trabalho_meio_periodo": [0],
    "nota_anterior": [75] 
})
previsao = modelo.predict(novo_aluno)

print("Nota prevista:", previsao[0])
# Fazer previsões utilizando apenas os 30% de teste
y_pred = modelo.predict(X_test)

# Mostrar algumas previsões
print("Valores reais:")
print(y_test.head())

print("\nValores previstos:")
print(y_pred[:5])
