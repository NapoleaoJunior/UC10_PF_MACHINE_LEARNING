import pandas as pd
import numpy as np
# Carregar CSV
df = pd.read_csv(
    r"desempenho_estudantil\student_performance_dataset.csv",
    sep=","
)
# Traduzir nomes das colunas
df.rename(columns={
    "student_id": "id_estudante",
    "gender": "genero",
    "study_time_hours": "horas_estudo",
    "attendance_percent": "percentual_frequencia",
    "sleep_hours": "horas_sono",
    "parental_education": "escolaridade_pais",
    "internet_access": "acesso_internet",
    "extracurricular_activities": "atividades_extracurriculares",
    "part_time_job": "trabalho_meio_periodo",
    "previous_grade": "nota_anterior",
    "final_exam_score": "pontuacao_prova_final",
    "final_grade": "nota_final"
}, inplace=True)
# Traduzir valores
df["genero"] = df["genero"].replace({
    "Male": "Masculino",
    "Female": "Feminino"
})
df["acesso_internet"] = df["acesso_internet"].replace({
    "Yes": "Sim",
    "No": "Não"
})
df["atividades_extracurriculares"] = df["atividades_extracurriculares"].replace({
    "Yes": "Sim",
    "No": "Não"
})
df["trabalho_meio_periodo"] = df["trabalho_meio_periodo"].replace({
    "Yes": "Sim",
    "No": "Não"
})
df["escolaridade_pais"] = df["escolaridade_pais"].replace({
    "High School": "Ensino Médio",
    "Bachelors": "Bacharelado",
    "Masters": "Mestrado",
    "PhD": "Doutorado"
})
df = df.replace(r'^\s*$', None, regex=True)
print(df.isnull().sum())
df.to_csv(
    r"desempenho_estudantil\traducao_dados.csv",
    index=False,
    encoding="utf-8-sig",
    na_rep="NULL"
)
print("Arquivo traduzido com sucesso!")
print(df.head())