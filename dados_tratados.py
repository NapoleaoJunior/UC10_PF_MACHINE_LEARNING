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

# ---------------------------------------------------------------
# Binarizar a nota final:  A, B, C = Aprovado (1)  |  D, F = Reprovado (0)
# ---------------------------------------------------------------
mapa_nota = {"A": 1, "B": 1, "C": 1, "D": 0, "F": 0}
df["nota_final"] = df["nota_final"].map(mapa_nota)

# Se quiser ser mais rígido (só A e B aprovam), use:
# mapa_nota = {"A": 1, "B": 1, "C": 0, "D": 0, "F": 0}

print("\nDistribuição de 'nota_final' APÓS binarização:")
print(df["nota_final"].value_counts())
print(f"Valores únicos: {sorted(df['nota_final'].dropna().unique())}")


# Salvar o novo arquivo tratado
df.to_csv(
    r"desempenho_estudantil\tratamento_dados_binario.csv",
    index=False,
    encoding="utf-8-sig",
    na_rep="0"
)

print("\nArquivo tratado com sucesso!")
print("\nPrimeiras linhas após a conversão:")
print(df.head())