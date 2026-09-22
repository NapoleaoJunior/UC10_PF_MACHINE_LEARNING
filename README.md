# Student Performance ML — Projeto Integrador

Previsão do desempenho acadêmico de estudantes (Aprovado/Reprovado) a partir
de hábitos de estudo e variáveis de estilo de vida, usando classificação
com Machine Learning.

## Dataset

[Student Performance & Study Habits Dataset — Kaggle](https://www.kaggle.com/datasets/harshadapatil31/student-performance-and-study-habits-dataset)

## Objetivo

Dado um conjunto de características de um estudante (horas de estudo, sono,
frequência, etc.), prever se ele será **Aprovado** ou **Reprovado**
(classificação binária), testando diferentes classificadores e configurações
de PCA para encontrar o melhor modelo.

## Pipeline

1. **Aquisição dos dados** — leitura do CSV com Pandas
2. **Análise exploratória** — estatísticas, gráficos, correlações
3. **Limpeza** — remoção de dados omissos e duplicados
4. **Balanceamento** — equalização das classes com SMOTE (aplicado só no treino)
5. **Treino/Teste** — split 80/20 estratificado
6. **Padronização** — StandardScaler
7. **PCA** — testado com 2, 3, 4 e 5 componentes
8. **Machine Learning** — KNN, Decision Tree e Random Forest (com GridSearchCV)
9. **Validação** — Accuracy, Precision, Recall, F1-Score
10. **Matriz de confusão** — para cada modelo
11. **Comparação dos modelos** — tabela e gráfico comparativo
12. **Melhor modelo** — selecionado por F1-macro
13. **Teste final** — avaliação no conjunto de teste nunca visto

## Como rodar

```bash
# 1. Criar ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Baixar o dataset do Kaggle e salvar em:
#    data/student_performance_dataset.csv

# 4. Rodar a pipeline, na ordem:
cd src
python 01_analise.py        # gera gráficos e mostra colunas reais do dataset
# --> ajuste config.py com o nome real da coluna alvo, se necessário
python 02_tratamento.py
python 03_balanceamento.py
python 04_modelos.py
python 05_avaliacao.py
```

## Estrutura do projeto

```
student-performance-ml/
├── data/                    # dataset bruto e processado
├── notebooks/                # EDA exploratória em Jupyter
├── src/                      # scripts da pipeline
│   ├── config.py
│   ├── 01_analise.py
│   ├── 02_tratamento.py
│   ├── 03_balanceamento.py
│   ├── 04_modelos.py
│   └── 05_avaliacao.py
├── models/                   # modelos treinados (.pkl)
├── reports/graficos/         # gráficos gerados (EDA, PCA, matriz de confusão)
├── requirements.txt
└── README.md
```

## Resultados

Após rodar `05_avaliacao.py`, a tabela comparativa fica em
`reports/comparacao_modelos.csv` e o melhor modelo salvo em
`models/melhor_modelo.pkl`.

> ⚠️ Se a acurácia média ficar abaixo de 50%, o script emite um alerta —
> sinal para revisar classificador, hiperparâmetros ou pré-processamento.

