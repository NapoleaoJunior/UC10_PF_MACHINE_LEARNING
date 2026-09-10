import pandas as pd

# Carregar o arquivo traduzido
df = pd.read_csv(
    r"desempenho_estudantil\traducao_dados.csv"
)

# Exibir as primeiras linhas
print("Dados originais:")
print(df.head())

# Converter variáveis categóricas para binárias/numéricas
df["genero"] = df["genero"].replace({
    "Masculino": 1,
    "Feminino": 0
})

df["escolaridade_pais"] = df["escolaridade_pais"].replace({
    "Ensino Médio": 1,
    "Bacharelado": 2,
    "Mestrado": 3,
    "Doutorado": 4
})

df["acesso_internet"] = df["acesso_internet"].replace({
    "Sim": 1,
    "Não": 0
})

df["atividades_extracurriculares"] = df["atividades_extracurriculares"].replace({
    "Sim": 1,
    "Não": 0
})

df["trabalho_meio_periodo"] = df["trabalho_meio_periodo"].replace({
    "Sim": 1,
    "Não": 0
})
df = df.replace(r'^\s*$', None, regex=True)
print(df.isnull().sum())
# Salvar o novo arquivo tratado
df.to_csv(
    r"desempenho_estudantil\tratamento_dados_binario.csv",
    index=False,
    encoding="utf-8-sig",
    na_rep="NULL"
)

print("\nArquivo tratado com sucesso!")
print("\nPrimeiras linhas após a conversão:")
print(df.head())