import pandas as pd
from sklearn.ensemble import RandomForestClassifier 

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

