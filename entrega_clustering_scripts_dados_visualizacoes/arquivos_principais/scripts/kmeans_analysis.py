import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Caminhos
preprocessed_dataset_path = "/home/ubuntu/preprocessed_highway_rail_crossing_dataset.csv"
results_dir = "/home/ubuntu/kmeans_results"

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

# --- Determinação do K Ótimo ---
print("\n--- Determinação do K Ótimo para K-Means Tradicional ---")

# Definir um range de K para testar
K_range = range(2, 16)

# 1. Método do Cotovelo (Elbow Method) - WCSS (Within-Cluster Sum of Squares)
wcss = [] # Também conhecido como inertia
for k_val in K_range:
    kmeans = KMeans(n_clusters=k_val, init="k-means++", random_state=42, n_init="auto")
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)
    print(f"K={k_val}, WCSS={kmeans.inertia_}")

plt.figure(figsize=(10, 6))
plt.plot(K_range, wcss, marker="o")
plt.title("Método do Cotovelo (WCSS) para K-Means Tradicional")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.xticks(list(K_range))
plt.grid(True)
elbow_plot_path = os.path.join(results_dir, "kmeans_elbow_method.png")
plt.savefig(elbow_plot_path)
plt.close()
print(f"Gráfico do Método do Cotovelo salvo em: {elbow_plot_path}")

# 2. Silhouette Score
silhouette_scores = []
for k_val in K_range:
    kmeans = KMeans(n_clusters=k_val, init="k-means++", random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(df_scaled)
    silhouette_avg = silhouette_score(df_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"K={k_val}, Silhouette Score={silhouette_avg}")

plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, marker="o")
plt.title("Silhouette Score para K-Means Tradicional")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("Silhouette Score Médio")
plt.xticks(list(K_range))
plt.grid(True)
silhouette_plot_path = os.path.join(results_dir, "kmeans_silhouette_scores.png")
plt.savefig(silhouette_plot_path)
plt.close()
print(f"Gráfico do Silhouette Score salvo em: {silhouette_plot_path}")

# 3. Davies-Bouldin Index
davies_bouldin_scores = []
for k_val in K_range:
    kmeans = KMeans(n_clusters=k_val, init="k-means++", random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(df_scaled)
    db_index = davies_bouldin_score(df_scaled, cluster_labels)
    davies_bouldin_scores.append(db_index)
    print(f"K={k_val}, Davies-Bouldin Index={db_index}")

plt.figure(figsize=(10, 6))
plt.plot(K_range, davies_bouldin_scores, marker="o")
plt.title("Davies-Bouldin Index para K-Means Tradicional")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("Davies-Bouldin Index")
plt.xticks(list(K_range))
plt.grid(True)
db_plot_path = os.path.join(results_dir, "kmeans_davies_bouldin_index.png")
plt.savefig(db_plot_path)
plt.close()
print(f"Gráfico do Davies-Bouldin Index salvo em: {db_plot_path}")

print("\n--- Análise das Métricas para Escolha do K Ótimo ---")

optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
print(f"K ótimo (baseado no maior Silhouette Score): {optimal_k_silhouette}")

optimal_k_db = K_range[np.argmin(davies_bouldin_scores)]
print(f"K ótimo (baseado no menor Davies-Bouldin Index): {optimal_k_db}")

K_OPTIMAL_PRELIMINAR = optimal_k_silhouette
print(f"K ótimo preliminar escolhido para visualização: {K_OPTIMAL_PRELIMINAR}")

# --- Executar K-Means com K Ótimo Preliminar e Visualizar ---
print(f"\n--- Executando K-Means com K = {K_OPTIMAL_PRELIMINAR} ---")
kmeans_final = KMeans(n_clusters=K_OPTIMAL_PRELIMINAR, init="k-means++", random_state=42, n_init="auto")
cluster_labels_final = kmeans_final.fit_predict(df_scaled)
df_scaled["Cluster_KMeans"] = cluster_labels_final

# Salvar o dataset com os labels do cluster
df_scaled.to_csv(os.path.join(results_dir, f"dataset_with_kmeans_clusters_k{K_OPTIMAL_PRELIMINAR}.csv"), index=False)
print(f"Dataset com labels do K-Means salvo em: {os.path.join(results_dir, f'dataset_with_kmeans_clusters_k{K_OPTIMAL_PRELIMINAR}.csv')}")

# Visualização dos clusters usando PCA para redução de dimensionalidade (2D)
pca = PCA(n_components=2, random_state=42)
df_pca = pca.fit_transform(df_scaled.drop("Cluster_KMeans", axis=1)) # Aplicar PCA nos dados originais escalados

plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=cluster_labels_final, cmap="viridis", alpha=0.7, edgecolors="k")
plt.title(f"Clusters K-Means (K={K_OPTIMAL_PRELIMINAR}) - Visualização PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend1)
plt.grid(True)
kmeans_pca_plot_path = os.path.join(results_dir, f"kmeans_clusters_pca_k{K_OPTIMAL_PRELIMINAR}.png")
plt.savefig(kmeans_pca_plot_path)
plt.close()
print(f"Visualização PCA dos clusters K-Means salva em: {kmeans_pca_plot_path}")

print("\nAnálise do K-Means tradicional (determinação de K e visualização inicial) concluída.")

