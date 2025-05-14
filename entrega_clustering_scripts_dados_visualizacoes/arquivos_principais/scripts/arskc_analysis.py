import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import sys

# Adicionar o diretório atual ao path para importar arskc_implementation
sys.path.append(os.getcwd())
from arskc_implementation import arskc # Assumindo que arskc_implementation.py está no mesmo diretório

# Caminhos
preprocessed_dataset_path = "/home/ubuntu/preprocessed_highway_rail_crossing_dataset.csv"
results_dir = "/home/ubuntu/arskc_results"

# Certificar que o diretório de resultados existe
os.makedirs(results_dir, exist_ok=True)

print(f"Carregando dataset pré-processado de {preprocessed_dataset_path}")
try:
    df_scaled = pd.read_csv(preprocessed_dataset_path)
    print(f"Dataset carregado. Dimensões: {df_scaled.shape}")
except FileNotFoundError:
    print(f"Erro: O arquivo {preprocessed_dataset_path} não foi encontrado.")
    sys.exit()
except Exception as e:
    print(f"Erro ao carregar o dataset: {e}")
    sys.exit()

# Parâmetros para ARSKC (exemplos, podem precisar de ajuste fino)
ALPHA_REG_ARSKC = 0.05 # Parâmetro de robustez (proporção de trimming)
BETA_REG_ARSKC = 0.05  # Parâmetro de esparsidade adaptativa
N_STARTS_ARSKC = 5     # Reduzido devido à complexidade, original do paper pode ser maior
MAX_ITER_ARS_ARSKC = 30 # Iterações do loop principal ARSKC
RANDOM_STATE = 42
EPSILON_W_ARSKC = 1e-6
EPSILON_V_ARSKC = 1e-6


# Definir um range de K para testar
K_range = range(2, 8) # Reduzido para ARSKC devido ao tempo de execução

print(f"\n--- Determinação do K Ótimo para ARSKC (alpha_reg={ALPHA_REG_ARSKC}, beta_reg={BETA_REG_ARSKC}) ---")

objective_arskc = []
silhouette_arskc = []
db_arskc = []

for k_val in K_range:
    print(f"  Testando K={k_val} para ARSKC...")
    arskc_result = arskc(df_scaled.copy(), K=k_val, 
                           alpha_reg=ALPHA_REG_ARSKC, beta_reg=BETA_REG_ARSKC,
                           n_starts=N_STARTS_ARSKC, max_iter_ars=MAX_ITER_ARS_ARSKC, 
                           epsilon_w=EPSILON_W_ARSKC, epsilon_v=EPSILON_V_ARSKC,
                           scaling=False, # Dados já estão escalados
                           random_state_seed=RANDOM_STATE, verbose=False)
    
    if arskc_result and "labels" in arskc_result and arskc_result.get("objective_value") is not None:
        labels = arskc_result["labels"]
        obj_val = arskc_result.get("objective_value", np.nan)
        # A implementação do ARSKC maximiza o objetivo (obj = -(loss + penalty))
        # Para o método do cotovelo, queremos ver onde o *aumento* no objetivo começa a diminuir.
        objective_arskc.append(obj_val if not (np.isnan(obj_val) or np.isinf(obj_val)) else (objective_arskc[-1] if objective_arskc and not (np.isnan(objective_arskc[-1]) or np.isinf(objective_arskc[-1])) else np.nan))
        
        # Garantir que os labels não sejam todos -1 (caso de falha total)
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and not (len(unique_labels) == 1 and unique_labels[0] == -1):
            silhouette_avg = silhouette_score(df_scaled, labels)
            db_index = davies_bouldin_score(df_scaled, labels)
        else:
            silhouette_avg = -1 # Penalidade se clustering falhar ou resultar em 1 cluster
            db_index = np.inf   # Penalidade
            
        silhouette_arskc.append(silhouette_avg)
        db_arskc.append(db_index)
        print(f"    K={k_val}, Obj_Val={obj_val:.4f}, Silhouette={silhouette_avg:.4f}, DB Index={db_index:.4f}")
        print(f"      Pesos de feature (não nulos): {np.sum(arskc_result.get('feature_weights', pd.Series(dtype='float64')) > EPSILON_W_ARSKC)}/{df_scaled.shape[1]}")
        print(f"      Observações trimadas (v=0): {np.sum(arskc_result.get('observation_weights', pd.Series(dtype='float64')) < EPSILON_V_ARSKC)}")
    else:
        print(f"    Falha ao executar ARSKC para K={k_val}. Resultado: {arskc_result}")
        objective_arskc.append(np.nan)
        silhouette_arskc.append(np.nan)
        db_arskc.append(np.nan)

# Plots para ARSKC
if any(not np.isnan(x) for x in objective_arskc):
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, objective_arskc, marker="o")
    plt.title(f"Método do Cotovelo (Valor Objetivo ARSKC) para alpha={ALPHA_REG_ARSKC}, beta={BETA_REG_ARSKC}")
    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("Valor Objetivo ARSKC (Maior é Melhor)")
    plt.xticks(list(K_range))
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"arskc_elbow_obj_alpha{ALPHA_REG_ARSKC}_beta{BETA_REG_ARSKC}.png"))
    plt.close()
    print(f"Gráfico Objetivo ARSKC salvo.")

if any(not np.isnan(x) for x in silhouette_arskc):
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, silhouette_arskc, marker="o")
    plt.title(f"Silhouette Score para ARSKC (alpha={ALPHA_REG_ARSKC}, beta={BETA_REG_ARSKC})")
    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("Silhouette Score Médio")
    plt.xticks(list(K_range))
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"arskc_silhouette_alpha{ALPHA_REG_ARSKC}_beta{BETA_REG_ARSKC}.png"))
    plt.close()
    print(f"Gráfico Silhouette ARSKC salvo.")

if any(not np.isnan(x) for x in db_arskc):
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, db_arskc, marker="o")
    plt.title(f"Davies-Bouldin Index para ARSKC (alpha={ALPHA_REG_ARSKC}, beta={BETA_REG_ARSKC})")
    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("Davies-Bouldin Index")
    plt.xticks(list(K_range))
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"arskc_db_index_alpha{ALPHA_REG_ARSKC}_beta{BETA_REG_ARSKC}.png"))
    plt.close()
    print(f"Gráfico Davies-Bouldin ARSKC salvo.")

# Escolher K ótimo preliminar para ARSKC
K_OPTIMAL_ARSKC = K_range[np.nanargmax(silhouette_arskc)] if any(not np.isnan(x) for x in silhouette_arskc) and not np.all(np.isnan(silhouette_arskc)) else K_range[0]
print(f"K ótimo preliminar para ARSKC (baseado no Silhouette): {K_OPTIMAL_ARSKC}")

# Executar ARSKC com K ótimo e visualizar
print(f"\n--- Executando ARSKC com K = {K_OPTIMAL_ARSKC} (alpha={ALPHA_REG_ARSKC}, beta={BETA_REG_ARSKC}) ---")
final_arskc_result = arskc(df_scaled.copy(), K=K_OPTIMAL_ARSKC, 
                             alpha_reg=ALPHA_REG_ARSKC, beta_reg=BETA_REG_ARSKC,
                             n_starts=N_STARTS_ARSKC, max_iter_ars=MAX_ITER_ARS_ARSKC,
                             epsilon_w=EPSILON_W_ARSKC, epsilon_v=EPSILON_V_ARSKC,
                             scaling=False, random_state_seed=RANDOM_STATE, verbose=True)

if final_arskc_result and "labels" in final_arskc_result and final_arskc_result.get("objective_value") is not None:
    df_scaled_arskc_clustered = df_scaled.copy()
    df_scaled_arskc_clustered["Cluster_ARSKC"] = final_arskc_result["labels"]
    # Salvar também os pesos das features e observações se disponíveis e úteis
    if "feature_weights" in final_arskc_result:
        pd.Series(final_arskc_result["feature_weights"], name="ARSKC_Feature_Weights").to_csv(os.path.join(results_dir, f"arskc_feature_weights_k{K_OPTIMAL_ARSKC}.csv"))
        print("Pesos das features do ARSKC salvos.")
    if "observation_weights" in final_arskc_result: # Estes são 0/1 para trimming
        df_scaled_arskc_clustered["ARSKC_Observation_Weight_v"] = final_arskc_result["observation_weights"]
    if "gamma_i_adaptive" in final_arskc_result: # Custo de robustez por observação
        df_scaled_arskc_clustered["ARSKC_Gamma_i_Adaptive"] = final_arskc_result["gamma_i_adaptive"]
    
    df_scaled_arskc_clustered.to_csv(os.path.join(results_dir, f"dataset_with_arskc_clusters_k{K_OPTIMAL_ARSKC}.csv"), index=False)
    print(f"Dataset com labels do ARSKC salvo.")

    # Visualização PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    df_pca_arskc = pca.fit_transform(df_scaled) 

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df_pca_arskc[:, 0], df_pca_arskc[:, 1], c=final_arskc_result["labels"], cmap="viridis", alpha=0.7, edgecolors="k")
    plt.title(f"Clusters ARSKC (K={K_OPTIMAL_ARSKC}, alpha={ALPHA_REG_ARSKC}, beta={BETA_REG_ARSKC}) - PCA")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    try:
        if len(np.unique(final_arskc_result["labels"])) > 1:
            legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
            plt.gca().add_artist(legend1)
    except Exception as e:
        print(f"Erro ao gerar legenda para PCA ARSKC: {e}")
        pass 
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"arskc_clusters_pca_k{K_OPTIMAL_ARSKC}.png"))
    plt.close()
    print(f"Visualização PCA dos clusters ARSKC salva.")
else:
    print(f"Falha ao executar ARSKC final. Resultado: {final_arskc_result}")

print("\nAnálise do ARSKC concluída.")
print(f"Lembrete: A implementação do ARSKC é uma tradução e pode necessitar de validação e refinamento adicionais em comparação com a implementação original em R e o paper.")

