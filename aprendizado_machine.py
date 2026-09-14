import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 

# Carregar CSV traduzido
df = pd.read_csv(r"desempenho_estudantil\tratamento_dados_binario.csv")
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
        "nota_anterior",
        "pontuacao_prova_final"
    ]
]
#variavel alvo
y = df["nota_final"]
#normaliza o espaço de caracteristicas
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)
#seperar os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# criar o modelo de Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
print(df["nota_final"].unique())
modelo.fit(X, y)
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
    "pontuacao_prova_final": [88]
})
previsao = modelo.predict(novo_aluno)

print("Nota prevista:", previsao[0])
