import pandas as pd
import numpy as np
import csv # Importar o módulo csv para usar suas constantes

# Caminho para o dataset
dataset_path = "/home/ubuntu/upload/Short-Highway-Rail-Crossing-Dataset.csv"

# Tentar carregar o dataset com diferentes estratégias
try:
    # Tentativa 1: engine python e on_bad_lines='warn'
    print("Tentando carregar com engine='python' e on_bad_lines='warn'...")
    df = pd.read_csv(dataset_path, engine='python', on_bad_lines='warn')
    print(f"Dataset carregado com sucesso usando engine='python'. Dimensões: {df.shape}")
    print("\nPrimeiras 5 linhas do dataset:")
    print(df.head())
except FileNotFoundError:
    print(f"Erro: O arquivo {dataset_path} não foi encontrado.")
    exit()
except Exception as e:
    print(f"Erro ao carregar o dataset com engine='python': {e}")
    try:
        # Tentativa 2: Especificar delimitador e quoting, ainda com on_bad_lines='warn'
        print("\nTentando carregar com delimitador=',' e quoting=csv.QUOTE_MINIMAL, on_bad_lines='warn'...")
        df = pd.read_csv(dataset_path, delimiter=',', quoting=csv.QUOTE_MINIMAL, on_bad_lines='warn', encoding='utf-8')
        print(f"Dataset carregado com sucesso usando delimitador e quoting. Dimensões: {df.shape}")
        print("\nPrimeiras 5 linhas do dataset:")
        print(df.head())
    except Exception as e2:
        print(f"Erro ao carregar o dataset com delimitador e quoting: {e2}")
        try:
            # Tentativa 3: Tentar com error_bad_lines=False (para versões mais antigas do pandas) ou on_bad_lines='skip'
            print("\nTentando carregar com on_bad_lines='skip'...")
            df = pd.read_csv(dataset_path, on_bad_lines='skip', encoding='utf-8')
            print(f"Dataset carregado com sucesso usando on_bad_lines='skip'. Linhas problemáticas foram puladas. Dimensões: {df.shape}")
            print("\nPrimeiras 5 linhas do dataset:")
            print(df.head())
        except Exception as e3:
            print(f"Falha em todas as tentativas de carregamento do CSV. Erro final: {e3}")
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
print(df.describe(include=["object"]))

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

eda_summary_path = "/home/ubuntu/eda_summary.txt"
original_dataset_snapshot_path = "/home/ubuntu/original_dataset_snapshot.csv"

with open(eda_summary_path, "w") as f:
    f.write(f"Dataset: {dataset_path}\n")
    f.write(f"Dimensões: {df.shape}\n\n")
    f.write("Primeiras 5 linhas:\n")
    f.write(str(df.head()) + "\n\n")
    
    # Capturar df.info() output
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    f.write("Informações do DataFrame:\n")
    f.write(info_str + "\n\n")
    
    f.write("Estatísticas Descritivas (Numéricas):\n")
    f.write(str(df.describe()) + "\n\n")
    f.write("Estatísticas Descritivas (Categóricas):\n")
    f.write(str(df.describe(include=["object"])) + "\n\n")
    f.write("Contagem de Valores Ausentes por Coluna:\n")
    f.write(str(missing_values) + "\n\n")
    f.write("Tipos de Dados das Colunas:\n")
    f.write(str(df.dtypes) + "\n\n")
    f.write("Nomes das Colunas:\n")
    f.write(str(df.columns.tolist()) + "\n")

print(f"\nResumo da EDA salvo em {eda_summary_path}")

# Salvar o dataframe original para referência, caso necessário mais tarde
df.to_csv(original_dataset_snapshot_path, index=False)
print(f"Snapshot do dataset original salvo em {original_dataset_snapshot_path}")

