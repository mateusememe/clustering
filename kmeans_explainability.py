import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import pickle

# Caminhos
preprocessed_dataset_path = "/home/ubuntu/preprocessed_highway_rail_crossing_dataset.csv"
kmeans_results_dir = "/home/ubuntu/kmeans_results"
explainability_dir = os.path.join(kmeans_results_dir, "explainability")
clustered_data_path = os.path.join(kmeans_results_dir, "dataset_with_kmeans_clusters_k5.csv") # Assumindo K=5 para K-Means

# Certificar que o diretório de resultados existe
os.makedirs(explainability_dir, exist_ok=True)

print(f"Carregando dataset pré-processado de {preprocessed_dataset_path}")
try:
    df_scaled = pd.read_csv(preprocessed_dataset_path)
    print(f"Dataset pré-processado carregado. Dimensões: {df_scaled.shape}")
except Exception as e:
    print(f"Erro ao carregar o dataset pré-processado: {e}")
    exit()

print(f"Carregando dataset com clusters K-Means de {clustered_data_path}")
try:
    df_clustered_kmeans = pd.read_csv(clustered_data_path)
    if "Cluster_KMeans" not in df_clustered_kmeans.columns:
        print("Erro: Coluna 'Cluster_KMeans' não encontrada no arquivo de dados clusterizados.")
        exit()
    kmeans_labels = df_clustered_kmeans["Cluster_KMeans"].values
    # Alinhar df_scaled com df_clustered_kmeans se necessário (assumindo que são o mesmo conjunto de dados em mesma ordem)
    if df_scaled.shape[0] != len(kmeans_labels):
        print("Erro: Disparidade no número de amostras entre dados escalados e labels do K-Means.")
        exit()
    print(f"Labels K-Means carregados. Número de clusters: {len(np.unique(kmeans_labels))}")
except Exception as e:
    print(f"Erro ao carregar o dataset com clusters K-Means: {e}")
    exit()

# Preparar dados para o classificador substituto
X = df_scaled.copy()
y = kmeans_labels
feature_names = X.columns.tolist()
class_names = [f"Cluster {i}" for i in sorted(np.unique(y))]

# Treinar um classificador substituto para explicar as atribuições de cluster
# Usar RandomForest como exemplo
print("\n--- Treinando Classificador Substituto para K-Means ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
surrogate_model = RandomForestClassifier(n_estimators=100, random_state=42)
surrogate_model.fit(X_train, y_train)
accuracy = surrogate_model.score(X_test, y_test)
print(f"Acurácia do modelo substituto (RandomForest) nos dados de teste: {accuracy:.4f}")

# Salvar o modelo substituto
model_path = os.path.join(explainability_dir, "kmeans_surrogate_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(surrogate_model, f)
print(f"Modelo substituto salvo em {model_path}")

# --- LIME para K-Means ---
print("\n--- Aplicando LIME para K-Means ---")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=class_names,
    mode="classification",
    random_state=42
)

# Explicar algumas instâncias (ex: uma de cada cluster do conjunto de teste)
num_instances_to_explain_lime = min(len(np.unique(y_test)), 5) # Explicar até 5 instâncias
explained_instances_indices_lime = []
for cluster_label in sorted(np.unique(y_test))[:num_instances_to_explain_lime]:
    idx = np.where(y_test == cluster_label)[0]
    if len(idx) > 0:
        explained_instances_indices_lime.append(idx[0])

for i, instance_idx in enumerate(explained_instances_indices_lime):
    instance = X_test.iloc[instance_idx]
    true_cluster = y_test[instance_idx]
    print(f"  Explicando instância {instance_idx} (do conjunto de teste) - Cluster Real: {true_cluster}")
    
    try:
        explanation_lime = lime_explainer.explain_instance(
            data_row=instance.values,
            predict_fn=surrogate_model.predict_proba,
            num_features=10, # Mostrar top 10 features
            top_labels=1 # Explicar o top label previsto
        )
        
        # Salvar a explicação LIME como HTML e plot
        lime_html_path = os.path.join(explainability_dir, f"kmeans_lime_instance_{instance_idx}_cluster{true_cluster}.html")
        explanation_lime.save_to_file(lime_html_path)
        print(f"    Explicação LIME (HTML) salva em: {lime_html_path}")

        fig = explanation_lime.as_pyplot_figure(label=explanation_lime.available_labels()[0])
        plt.title(f"LIME para Instância {instance_idx} - Previsto para Cluster {explanation_lime.available_labels()[0]} (Real: {true_cluster})")
        plt.tight_layout()
        lime_plot_path = os.path.join(explainability_dir, f"kmeans_lime_instance_{instance_idx}_cluster{true_cluster}.png")
        plt.savefig(lime_plot_path)
        plt.close(fig)
        print(f"    Plot LIME salvo em: {lime_plot_path}")

    except Exception as e:
        print(f"    Erro ao gerar explicação LIME para instância {instance_idx}: {e}")

# --- SHAP para K-Means ---
print("\n--- Aplicando SHAP para K-Means ---")
# Usar KernelExplainer se o modelo não for baseado em árvore ou se quisermos ser model-agnostic
# Para RandomForest, TreeExplainer é mais eficiente.

# Para TreeExplainer, precisamos do modelo e dos dados
shap_explainer = shap.TreeExplainer(surrogate_model)
# Calcular SHAP values (pode ser demorado para datasets grandes, usar uma amostra)
sample_X_shap = shap.sample(X_test, 100) if len(X_test) > 100 else X_test
shap_values = shap_explainer.shap_values(sample_X_shap) # Retorna lista de arrays (um por classe)

# Plot SHAP Summary (Importância global das features)
# Para classificação multi-classe, shap_values é uma lista de arrays.
# Podemos plotar para cada classe ou um sumário agregado.

# Sumário para a primeira classe como exemplo (ou média sobre as classes)
plt.figure(figsize=(10,8))
shap.summary_plot(shap_values[0], sample_X_shap, feature_names=feature_names, show=False)
plt.title(f"SHAP Summary Plot para K-Means (Exemplo: Cluster {class_names[0]})")
plt.tight_layout()
shap_summary_path = os.path.join(explainability_dir, f"kmeans_shap_summary_cluster0.png")
plt.savefig(shap_summary_path)
plt.close()
print(f"Plot SHAP Summary (Cluster 0) salvo em: {shap_summary_path}")

# Plot SHAP Summary (beeswarm) para todas as classes
for i, class_name in enumerate(class_names):
    try:
        plt.figure(figsize=(10, max(8, len(feature_names)//2)))
        shap.summary_plot(shap_values[i], sample_X_shap, feature_names=feature_names, show=False, plot_type="bar")
        plt.title(f"SHAP Feature Importance para K-Means (Cluster {class_name.split()[-1]})")
        plt.tight_layout()
        shap_summary_bar_path = os.path.join(explainability_dir, f"kmeans_shap_summary_bar_cluster{class_name.split()[-1]}.png")
        plt.savefig(shap_summary_bar_path)
        plt.close()
        print(f"  Plot SHAP Summary Bar (Cluster {class_name.split()[-1]}) salvo em: {shap_summary_bar_path}")

        plt.figure(figsize=(12,8))
        shap.summary_plot(shap_values[i], sample_X_shap, feature_names=feature_names, show=False, plot_type="dot") # beeswarm
        plt.title(f"SHAP Beeswarm Plot para K-Means (Cluster {class_name.split()[-1]})")
        plt.tight_layout()
        shap_beeswarm_path = os.path.join(explainability_dir, f"kmeans_shap_beeswarm_cluster{class_name.split()[-1]}.png")
        plt.savefig(shap_beeswarm_path)
        plt.close()
        print(f"  Plot SHAP Beeswarm (Cluster {class_name.split()[-1]}) salvo em: {shap_beeswarm_path}")

    except Exception as e:
        print(f"    Erro ao gerar plots SHAP para Cluster {class_name.split()[-1]}: {e}")

# SHAP Dependence Plots para algumas features importantes
# Identificar features importantes (ex: do summary plot)
if hasattr(surrogate_model, 'feature_importances_'):
    importances = surrogate_model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    top_n_features = 5

    for i in range(min(top_n_features, len(feature_names))):
        feature_idx = sorted_indices[i]
        feature_name = feature_names[feature_idx]
        try:
            plt.figure()
            # Para classificadores multi-classe, shap_values é uma lista.
            # Dependence plot pode ser feito para uma classe específica ou interação.
            shap.dependence_plot(feature_name, shap_values[0], sample_X_shap, feature_names=feature_names, interaction_index="auto", show=False)
            plt.title(f"SHAP Dependence Plot: {feature_name} (para Cluster {class_names[0]})")
            plt.tight_layout()
            shap_dependence_path = os.path.join(explainability_dir, f"kmeans_shap_dependence_{feature_name}_cluster0.png")
            plt.savefig(shap_dependence_path)
            plt.close()
            print(f"  Plot SHAP Dependence ({feature_name}, Cluster 0) salvo em: {shap_dependence_path}")
        except Exception as e:
            print(f"    Erro ao gerar SHAP Dependence Plot para {feature_name}: {e}")

print("\nAnálise de explicabilidade LIME e SHAP para K-Means concluída.")

