import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import sys

# Adicionar o diretório atual ao path para importar rskc_implementation
sys.path.append(os.getcwd()) 
from rskc_implementation import rskc # Assumindo que rskc_implementation.py está no mesmo diretório ou no PYTHONPATH

# Caminhos
preprocessed_dataset_path = "/home/ubuntu/preprocessed_highway_rail_crossing_dataset.csv"
results_dir = "/home/ubuntu/rskc_results"

# Certificar que o diretório de resultados existe
os.makedirs(results_dir, exist_ok=True)

print(f"Carregando dataset pré-processado de {preprocessed_dataset_path}")
try:
    df_scaled = pd.read_csv(preprocessed_dataset_path)
    print(f"Dataset carregado. Dimensões: {df_scaled.shape}")
except FileNotFoundError:
    print(f"Erro: O arquivo {preprocessed_dataset_path} não foi encontrado.")
    exit()
except Exception as e:
    print(f"Erro ao carregar o dataset: {e}")
    exit()

# Parâmetros para RSKC
ALPHA_TRIM = 0.05 # Proporção de trimming, exemplo. Pode ser ajustado.
L1_PARAM_SPARSE = 5 # Exemplo de L1 para o caso esparso. Ajustar conforme necessário.
N_START_RSKC = 50 # Reduzido para RSKC devido à complexidade, original era 200 para K-Means simples.
RANDOM_STATE = 42

# Definir um range de K para testar
K_range = range(2, 8) # Reduzido para RSKC devido ao tempo de execução e complexidade

# --- Análise para RSKC Não Esparso (Trimmed K-Means like) ---
print(f"\n--- Determinação do K Ótimo para RSKC Não Esparso (alpha={ALPHA_TRIM}, L1=None) ---")

wcss_rskc_nonsparse = []
silhouette_rskc_nonsparse = []
db_rskc_nonsparse = []

for k_val in K_range:
    print(f"  Testando K={k_val} para RSKC Não Esparso...")
    # L1=None para o caso não esparso (Trimmed K-Means)
    rskc_result = rskc(df_scaled.copy(), ncl=k_val, alpha=ALPHA_TRIM, L1=None, 
                         nstart=N_START_RSKC, silent=True, scaling=False, random_state=RANDOM_STATE)
    
    if rskc_result and rskc_result.get("labels") is not None:
        labels = rskc_result["labels"]
        # WCSS (ou WWSS para o caso não esparso) é retornado pelo rskc
        wwss = rskc_result.get("WWSS", np.nan)
        wcss_rskc_nonsparse.append(wwss if not np.isnan(wwss) else (wcss_rskc_nonsparse[-1] if wcss_rskc_nonsparse else np.nan) ) # Fallback se NaN
        
        if len(np.unique(labels)) > 1: # Silhouette e DB precisam de pelo menos 2 clusters
            silhouette_avg = silhouette_score(df_scaled, labels)
            db_index = davies_bouldin_score(df_scaled, labels)
        else:
            silhouette_avg = -1 # Penalidade
            db_index = np.inf   # Penalidade
            
        silhouette_rskc_nonsparse.append(silhouette_avg)
        db_rskc_nonsparse.append(db_index)
        print(f"    K={k_val}, WWSS={wwss:.2f}, Silhouette={silhouette_avg:.4f}, DB Index={db_index:.4f}")
    else:
        print(f"    Falha ao executar RSKC Não Esparso para K={k_val}")
        wcss_rskc_nonsparse.append(np.nan)
        silhouette_rskc_nonsparse.append(np.nan)
        db_rskc_nonsparse.append(np.nan)

# Plots para RSKC Não Esparso
if any(not np.isnan(x) for x in wcss_rskc_nonsparse):
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, wcss_rskc_nonsparse, marker="o")
    plt.title(f"Método do Cotovelo (WWSS) para RSKC Não Esparso (alpha={ALPHA_TRIM})")
    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("WWSS (Within-Cluster Sum of Squares)")
    plt.xticks(list(K_range))
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"rskc_nonsparse_elbow_alpha{ALPHA_TRIM}.png"))
    plt.close()
    print(f"Gráfico WWSS RSKC Não Esparso salvo.")

if any(not np.isnan(x) for x in silhouette_rskc_nonsparse):
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, silhouette_rskc_nonsparse, marker="o")
    plt.title(f"Silhouette Score para RSKC Não Esparso (alpha={ALPHA_TRIM})")
    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("Silhouette Score Médio")
    plt.xticks(list(K_range))
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"rskc_nonsparse_silhouette_alpha{ALPHA_TRIM}.png"))
    plt.close()
    print(f"Gráfico Silhouette RSKC Não Esparso salvo.")

if any(not np.isnan(x) for x in db_rskc_nonsparse):
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, db_rskc_nonsparse, marker="o")
    plt.title(f"Davies-Bouldin Index para RSKC Não Esparso (alpha={ALPHA_TRIM})")
    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("Davies-Bouldin Index")
    plt.xticks(list(K_range))
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"rskc_nonsparse_db_index_alpha{ALPHA_TRIM}.png"))
    plt.close()
    print(f"Gráfico Davies-Bouldin RSKC Não Esparso salvo.")

# Escolher K ótimo preliminar para RSKC Não Esparso
K_OPTIMAL_RSKC_NONSPARSE = K_range[np.nanargmax(silhouette_rskc_nonsparse)] if any(not np.isnan(x) for x in silhouette_rskc_nonsparse) else K_range[0]
print(f"K ótimo preliminar para RSKC Não Esparso (baseado no Silhouette): {K_OPTIMAL_RSKC_NONSPARSE}")

# Executar RSKC Não Esparso com K ótimo e visualizar
print(f"\n--- Executando RSKC Não Esparso com K = {K_OPTIMAL_RSKC_NONSPARSE} (alpha={ALPHA_TRIM}) ---")
final_rskc_nonsparse_result = rskc(df_scaled.copy(), ncl=K_OPTIMAL_RSKC_NONSPARSE, alpha=ALPHA_TRIM, L1=None,
                                     nstart=N_START_RSKC, silent=False, scaling=False, random_state=RANDOM_STATE)

if final_rskc_nonsparse_result and final_rskc_nonsparse_result.get("labels") is not None:
    df_scaled_nonsparse_clustered = df_scaled.copy()
    df_scaled_nonsparse_clustered["Cluster_RSKC_NonSparse"] = final_rskc_nonsparse_result["labels"]
    df_scaled_nonsparse_clustered.to_csv(os.path.join(results_dir, f"dataset_with_rskc_nonsparse_clusters_k{K_OPTIMAL_RSKC_NONSPARSE}.csv"), index=False)
    print(f"Dataset com labels do RSKC Não Esparso salvo.")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    df_pca_nonsparse = pca.fit_transform(df_scaled) # PCA nos dados originais escalados

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df_pca_nonsparse[:, 0], df_pca_nonsparse[:, 1], c=final_rskc_nonsparse_result["labels"], cmap="viridis", alpha=0.7, edgecolors="k")
    plt.title(f"Clusters RSKC Não Esparso (K={K_OPTIMAL_RSKC_NONSPARSE}, alpha={ALPHA_TRIM}) - PCA")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    try:
        legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.gca().add_artist(legend1)
    except Exception:
        pass # Em caso de problema com legend_elements se houver apenas 1 cluster
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"rskc_nonsparse_clusters_pca_k{K_OPTIMAL_RSKC_NONSPARSE}.png"))
    plt.close()
    print(f"Visualização PCA dos clusters RSKC Não Esparso salva.")
else:
    print("Falha ao executar RSKC Não Esparso final.")


# --- Análise para RSKC Esparso --- 
# AVISO: A implementação de _solve_for_weights é um placeholder, então a esparsidade não será ótima.
print(f"\n--- Determinação do K Ótimo para RSKC Esparso (alpha={ALPHA_TRIM}, L1={L1_PARAM_SPARSE}) ---")
print("AVISO: A esparsidade é baseada em uma implementação placeholder de _solve_for_weights.")

wcss_rskc_sparse = [] # Usaremos WBSS se disponível e significativo
silhouette_rskc_sparse = []
db_rskc_sparse = []

for k_val in K_range:
    print(f"  Testando K={k_val} para RSKC Esparso...")
    rskc_result_sparse = rskc(df_scaled.copy(), ncl=k_val, alpha=ALPHA_TRIM, L1=L1_PARAM_SPARSE, 
                                nstart=N_START_RSKC, silent=True, scaling=False, random_state=RANDOM_STATE)
    
    if rskc_result_sparse and rskc_result_sparse.get("labels") is not None:
        labels_sparse = rskc_result_sparse["labels"]
        wbss = rskc_result_sparse.get("WBSS", np.nan) # WBSS é o objetivo para o esparso
        # Se WBSS for NaN ou inf, usar um fallback ou ignorar para o gráfico do cotovelo
        wcss_rskc_sparse.append(wbss if not (np.isnan(wbss) or np.isinf(wbss)) else (wcss_rskc_sparse[-1] if wcss_rskc_sparse and not (np.isnan(wcss_rskc_sparse[-1]) or np.isinf(wcss_rskc_sparse[-1])) else np.nan))

        if len(np.unique(labels_sparse)) > 1:
            silhouette_avg_sparse = silhouette_score(df_scaled, labels_sparse)
            db_index_sparse = davies_bouldin_score(df_scaled, labels_sparse)
        else:
            silhouette_avg_sparse = -1
            db_index_sparse = np.inf
            
        silhouette_rskc_sparse.append(silhouette_avg_sparse)
        db_rskc_sparse.append(db_index_sparse)
        print(f"    K={k_val}, WBSS={wbss:.2f}, Silhouette={silhouette_avg_sparse:.4f}, DB Index={db_index_sparse:.4f}")
        print(f"      Pesos (sum={np.sum(rskc_result_sparse.get('weights', [])):.2f}): {rskc_result_sparse.get('weights')}")
    else:
        print(f"    Falha ao executar RSKC Esparso para K={k_val}")
        wcss_rskc_sparse.append(np.nan)
        silhouette_rskc_sparse.append(np.nan)
        db_rskc_sparse.append(np.nan)

# Plots para RSKC Esparso
if any(not np.isnan(x) for x in wcss_rskc_sparse):
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, wcss_rskc_sparse, marker="o")
    plt.title(f"Objetivo (WBSS) para RSKC Esparso (alpha={ALPHA_TRIM}, L1={L1_PARAM_SPARSE})")
    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("WBSS (Weighted Between Sum of Squares)")
    plt.xticks(list(K_range))
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"rskc_sparse_wbss_alpha{ALPHA_TRIM}_L1_{L1_PARAM_SPARSE}.png"))
    plt.close()
    print(f"Gráfico WBSS RSKC Esparso salvo.")

if any(not np.isnan(x) for x in silhouette_rskc_sparse):
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, silhouette_rskc_sparse, marker="o")
    plt.title(f"Silhouette Score para RSKC Esparso (alpha={ALPHA_TRIM}, L1={L1_PARAM_SPARSE})")
    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("Silhouette Score Médio")
    plt.xticks(list(K_range))
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"rskc_sparse_silhouette_alpha{ALPHA_TRIM}_L1_{L1_PARAM_SPARSE}.png"))
    plt.close()
    print(f"Gráfico Silhouette RSKC Esparso salvo.")

if any(not np.isnan(x) for x in db_rskc_sparse):
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, db_rskc_sparse, marker="o")
    plt.title(f"Davies-Bouldin Index para RSKC Esparso (alpha={ALPHA_TRIM}, L1={L1_PARAM_SPARSE})")
    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("Davies-Bouldin Index")
    plt.xticks(list(K_range))
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"rskc_sparse_db_index_alpha{ALPHA_TRIM}_L1_{L1_PARAM_SPARSE}.png"))
    plt.close()
    print(f"Gráfico Davies-Bouldin RSKC Esparso salvo.")

# Escolher K ótimo preliminar para RSKC Esparso
K_OPTIMAL_RSKC_SPARSE = K_range[np.nanargmax(silhouette_rskc_sparse)] if any(not np.isnan(x) for x in silhouette_rskc_sparse) else K_range[0]
print(f"K ótimo preliminar para RSKC Esparso (baseado no Silhouette): {K_OPTIMAL_RSKC_SPARSE}")

# Executar RSKC Esparso com K ótimo e visualizar
print(f"\n--- Executando RSKC Esparso com K = {K_OPTIMAL_RSKC_SPARSE} (alpha={ALPHA_TRIM}, L1={L1_PARAM_SPARSE}) ---")
final_rskc_sparse_result = rskc(df_scaled.copy(), ncl=K_OPTIMAL_RSKC_SPARSE, alpha=ALPHA_TRIM, L1=L1_PARAM_SPARSE,
                                  nstart=N_START_RSKC, silent=False, scaling=False, random_state=RANDOM_STATE)

if final_rskc_sparse_result and final_rskc_sparse_result.get("labels") is not None:
    df_scaled_sparse_clustered = df_scaled.copy()
    df_scaled_sparse_clustered["Cluster_RSKC_Sparse"] = final_rskc_sparse_result["labels"]
    df_scaled_sparse_clustered.to_csv(os.path.join(results_dir, f"dataset_with_rskc_sparse_clusters_k{K_OPTIMAL_RSKC_SPARSE}.csv"), index=False)
    print(f"Dataset com labels do RSKC Esparso salvo.")
    print(f"Pesos finais (Esparso): {final_rskc_sparse_result.get('weights')}")

    pca_sparse = PCA(n_components=2, random_state=RANDOM_STATE)
    df_pca_sparse = pca_sparse.fit_transform(df_scaled) 

    plt.figure(figsize=(12, 8))
    scatter_sparse = plt.scatter(df_pca_sparse[:, 0], df_pca_sparse[:, 1], c=final_rskc_sparse_result["labels"], cmap="viridis", alpha=0.7, edgecolors="k")
    plt.title(f"Clusters RSKC Esparso (K={K_OPTIMAL_RSKC_SPARSE}, alpha={ALPHA_TRIM}, L1={L1_PARAM_SPARSE}) - PCA")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    try:
        legend_sparse = plt.legend(*scatter_sparse.legend_elements(), title="Clusters")
        plt.gca().add_artist(legend_sparse)
    except Exception:
        pass 
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"rskc_sparse_clusters_pca_k{K_OPTIMAL_RSKC_SPARSE}.png"))
    plt.close()
    print(f"Visualização PCA dos clusters RSKC Esparso salva.")
else:
    print("Falha ao executar RSKC Esparso final.")

print("\nAnálise do RSKC (Não Esparso e Esparso com implementação placeholder) concluída.")

