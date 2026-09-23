import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# Carregar CSV traduzido
df = pd.read_csv(r"desempenho_estudantil\tratamento_dados_binario.csv")
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
# normaliza o espaço de caracteristicas
# seperar os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# criar o modelo de Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
print(df["nota_final"].unique())
# criar o modelo de Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)

# treinar com 70%
modelo.fit(X_train, y_train)

# testar com 30%
y_pred = modelo.predict(X_test)

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
novo_aluno = novo_aluno[X.columns]
previsao = modelo.predict(novo_aluno)
print("Previsão:", previsao)
if previsao[0] == 1:
    print("Aluno APROVADO ✅")
else:
    print("Aluno REPROVADO ❌")
print("Nota prevista:", previsao[0])
print("nota_final:", modelo.predict_proba(novo_aluno))
# Fazer previsões utilizando apenas os 30% de teste
y_pred = modelo.predict(X_test)
# Mostrar algumas previsões
print("Valores reais:")
print(y_test.head())
print("\nValores previstos:")
print(y_pred[:5])
