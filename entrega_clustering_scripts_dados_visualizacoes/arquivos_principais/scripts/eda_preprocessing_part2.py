import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Caminho para o dataset carregado anteriormente
dataset_path = "/home/ubuntu/original_dataset_snapshot.csv"
preprocessed_dataset_path = "/home/ubuntu/preprocessed_highway_rail_crossing_dataset.csv"

print(f"Carregando dataset de {dataset_path}")
try:
    df = pd.read_csv(dataset_path, low_memory=False) # low_memory=False to handle mixed types if any
    print(f"Dataset carregado. Dimensões: {df.shape}")
except FileNotFoundError:
    print(f"Erro: O arquivo {dataset_path} não foi encontrado.")
    exit()
except Exception as e:
    print(f"Erro ao carregar o dataset: {e}")
    exit()

# --- 2. Seleção de Features e Pré-processamento Detalhado ---
print("\n--- Seleção de Features e Pré-processamento Detalhado ---")

# Features selecionadas com base na solicitação do usuário e análise exploratória
# Foco: prejuízo financeiro, severidade, características do incidente/travessia
numeric_features = [
    'Vehicle Damage Cost',       # Prejuízo financeiro
    'Total Killed Form 57',      # Severidade
    'Total Injured Form 57',     # Severidade
    'Train Speed',               # Característica do incidente
    'Estimated Vehicle Speed'    # Característica do incidente
]

categorical_features = [
    'Hazmat Involvement Code',   # Severidade (proxy)
    'Highway User Code',         # Característica do incidente
    'Weather Condition Code',    # Contexto
    'Visibility Code',           # Contexto
    'Public/Private Code',       # Característica da travessia
    'Equipment Struck Code'      # Característica do incidente
]

selected_features = numeric_features + categorical_features
df_selected = df[selected_features].copy()
print(f"\nShape do DataFrame após seleção de features: {df_selected.shape}")
print(f"Features selecionadas: {selected_features}")

# --- Tratamento de Valores Ausentes e Tipos de Dados ---
print("\n--- Tratamento de Valores Ausentes e Tipos de Dados ---")

# Para colunas numéricas:
# Converter para numérico (coercing errors to NaN) e imputar com a mediana
for col in numeric_features:
    if col in df_selected.columns:
        df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')
        median_val = df_selected[col].median()
        df_selected[col].fillna(median_val, inplace=True)
        print(f"Coluna numérica '{col}': NAs preenchidos com mediana ({median_val}).")
    else:
        print(f"Atenção: Coluna numérica '{col}' não encontrada no DataFrame.")

# Para colunas categóricas:
# Converter para string (para garantir tratamento categórico) e imputar com a moda
for col in categorical_features:
    if col in df_selected.columns:
        df_selected[col] = df_selected[col].astype(str) # Garante que é string
        mode_val = df_selected[col].mode()[0] # Pega o primeiro modo se houver múltiplos
        df_selected[col].fillna(mode_val, inplace=True)
        # Também preencher 'nan' (string) que pode ter surgido da conversão .astype(str) de NaNs originais
        df_selected[col].replace('nan', mode_val, inplace=True)
        print(f"Coluna categórica '{col}': NAs (e strings 'nan') preenchidos com moda ('{mode_val}').")
    else:
        print(f"Atenção: Coluna categórica '{col}' não encontrada no DataFrame.")

print("\nVerificação de valores ausentes após imputação:")
print(df_selected.isnull().sum())

# --- Codificação de Variáveis Categóricas (One-Hot Encoding) ---
print("\n--- Codificação de Variáveis Categóricas ---")
df_processed = pd.get_dummies(df_selected, columns=categorical_features, dummy_na=False, dtype=int)
print(f"Shape do DataFrame após One-Hot Encoding: {df_processed.shape}")
print("Primeiras 5 linhas do DataFrame processado (antes da normalização):")
print(df_processed.head())

# --- Normalização/Padronização das Features ---
print("\n--- Normalização/Padronização (StandardScaler) ---")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_processed)
df_scaled = pd.DataFrame(scaled_features, columns=df_processed.columns)

print("Primeiras 5 linhas do DataFrame normalizado/padronizado:")
print(df_scaled.head())

# --- Salvar o Dataset Pré-processado ---
print(f"\nSalvando dataset pré-processado em: {preprocessed_dataset_path}")
df_scaled.to_csv(preprocessed_dataset_path, index=False)
print("Dataset pré-processado salvo com sucesso.")

# Informações finais
print("\n--- Resumo do Pré-processamento ---")
print(f"Dataset original shape: {df.shape}")
print(f"Dataset selecionado shape: {df_selected.shape}")
print(f"Dataset após one-hot encoding shape: {df_processed.shape}")
print(f"Dataset final pré-processado e normalizado shape: {df_scaled.shape}")
print(f"Colunas no dataset final: {df_scaled.columns.tolist()}")

print("\nPré-processamento concluído.")

