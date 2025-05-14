import pandas as pd
import numpy as np

# Caminho para o dataset
dataset_path = "/home/ubuntu/upload/Short-Highway-Rail-Crossing-Dataset.csv"

# Carregar o dataset
try:
    df = pd.read_csv(dataset_path)
    print(f"Dataset carregado com sucesso. Dimensões: {df.shape}")
    print("\nPrimeiras 5 linhas do dataset:")
    print(df.head())
except FileNotFoundError:
    print(f"Erro: O arquivo {dataset_path} não foi encontrado.")
    exit()
except Exception as e:
    print(f"Erro ao carregar o dataset: {e}")
    exit()

# --- 1. Análise Exploratória dos Dados (EDA) --- 
print("\n--- Análise Exploratória dos Dados (EDA) ---")

# Informações básicas do dataset
print("\nInformações do DataFrame:")
df.info()

# Estatísticas descritivas para colunas numéricas
print("\nEstatísticas Descritivas (Numéricas):")
print(df.describe())

# Estatísticas descritivas para colunas categóricas (object)
print("\nEstatísticas Descritivas (Categóricas):")
print(df.describe(include=['object']))

# Verificar valores ausentes
print("\nContagem de Valores Ausentes por Coluna:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
if missing_values.sum() == 0:
    print("Não há valores ausentes no dataset.")

# Identificar tipos de dados únicos
print("\nTipos de Dados das Colunas:")
print(df.dtypes)

# Listar todas as colunas para facilitar a seleção de features
print("\nNomes das Colunas:")
print(df.columns.tolist())

# --- Fim da Análise Exploratória Inicial ---
# Próximos passos seriam a seleção de features e pré-processamento mais aprofundado.
# Por enquanto, vamos salvar um resumo da EDA em um arquivo.

eda_summary = []
eda_summary.append(f"Dataset: {dataset_path}")
eda_summary.append(f"Dimensões: {df.shape}")
eda_summary.append("\nPrimeiras 5 linhas:")
eda_summary.append(str(df.head()))
eda_summary.append("\nInformações do DataFrame:")
eda_summary.append(str(df.info(verbose=True, buf=None)))
eda_summary.append("\nEstatísticas Descritivas (Numéricas):")
eda_summary.append(str(df.describe()))
eda_summary.append("\nEstatísticas Descritivas (Categóricas):")
eda_summary.append(str(df.describe(include=['object'])))
eda_summary.append("\nContagem de Valores Ausentes por Coluna:")
eda_summary.append(str(missing_values))
eda_summary.append("\nTipos de Dados das Colunas:")
eda_summary.append(str(df.dtypes))
eda_summary.append("\nNomes das Colunas:")
eda_summary.append(str(df.columns.tolist()))

with open("/home/ubuntu/eda_summary.txt", "w") as f:
    for item in eda_summary:
        f.write(f"{item}\n\n")
print("\nResumo da EDA salvo em /home/ubuntu/eda_summary.txt")

# Salvar o dataframe original para referência, caso necessário mais tarde
df.to_csv("/home/ubuntu/original_dataset_snapshot.csv", index=False)
print("Snapshot do dataset original salvo em /home/ubuntu/original_dataset_snapshot.csv")

