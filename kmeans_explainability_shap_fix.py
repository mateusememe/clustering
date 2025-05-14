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

print("\n--- Treinando Classificador Substituto para K-Means ---")
X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
surrogate_model = RandomForestClassifier(n_estimators=100, random_state=42)
# Treinar com nomes de features para que o modelo os armazene se possível
surrogate_model.fit(X_train_df, y_train)
accuracy = surrogate_model.score(X_test_df, y_test)
print(f"Acurácia do modelo substituto (RandomForest) nos dados de teste: {accuracy:.4f}")

model_path = os.path.join(explainability_dir, "kmeans_surrogate_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(surrogate_model, f)
print(f"Modelo substituto salvo em {model_path}")

# --- LIME para K-Means ---
print("\n--- Aplicando LIME para K-Means ---")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_df.values, # LIME espera numpy array para training_data
    feature_names=feature_names,
    class_names=class_names,
    mode="classification",
    random_state=42
)

num_instances_to_explain_lime = min(len(np.unique(y_test)), 5)
explained_instances_indices_lime = []
for cluster_label in sorted(np.unique(y_test))[:num_instances_to_explain_lime]:
    idx = np.where(y_test == cluster_label)[0]
    if len(idx) > 0:
        explained_instances_indices_lime.append(idx[0])

for i, instance_idx_in_test in enumerate(explained_instances_indices_lime):
    instance_series = X_test_df.iloc[instance_idx_in_test]
    true_cluster = y_test[instance_idx_in_test]
    # Obter o índice original da instância no dataset X completo, se X_test_df tem reset_index implícito ou não.
    # Para simplicidade, vamos usar o índice dentro de X_test_df para nomear arquivos.
    print(f"  Explicando instância com índice {instance_idx_in_test} em X_test_df - Cluster Real: {true_cluster}")
    
    try:
        explanation_lime = lime_explainer.explain_instance(
            data_row=instance_series.values, # LIME espera numpy array para data_row
            predict_fn=surrogate_model.predict_proba,
            num_features=10,
            top_labels=1
        )
        
        lime_html_path = os.path.join(explainability_dir, f"kmeans_lime_instance_testidx{instance_idx_in_test}_cluster{true_cluster}.html")
        explanation_lime.save_to_file(lime_html_path)
        print(f"    Explicação LIME (HTML) salva em: {lime_html_path}")

        fig = explanation_lime.as_pyplot_figure(label=explanation_lime.available_labels()[0])
        plt.title(f"LIME para Instância (test_idx {instance_idx_in_test}) - Prev. Cluster {explanation_lime.available_labels()[0]} (Real: {true_cluster})")
        plt.tight_layout()
        lime_plot_path = os.path.join(explainability_dir, f"kmeans_lime_instance_testidx{instance_idx_in_test}_cluster{true_cluster}.png")
        plt.savefig(lime_plot_path)
        plt.close(fig)
        print(f"    Plot LIME salvo em: {lime_plot_path}")

    except Exception as e:
        print(f"    Erro ao gerar explicação LIME para instância com índice {instance_idx_in_test} em X_test_df: {e}")

# --- SHAP para K-Means ---
print("\n--- Aplicando SHAP para K-Means ---")
shap_explainer = shap.TreeExplainer(surrogate_model)
# Usar X_test_df para sample_X_shap_df para consistência com o que o modelo foi avaliado
sample_X_shap_df = shap.sample(X_test_df, 100) if len(X_test_df) > 100 else X_test_df
sample_X_shap_np = sample_X_shap_df.values # Convert to NumPy array for SHAP functions if needed

# Calcular SHAP values usando o array NumPy
shap_values = shap_explainer.shap_values(sample_X_shap_np)

if not isinstance(shap_values, list):
    print("SHAP values não são uma lista como esperado para multi-classe. Verifique o explainer.")
    exit()
if not all(isinstance(arr, np.ndarray) for arr in shap_values):
    print("Nem todos os elementos em shap_values são arrays NumPy.")
    exit()

print(f"Número de classes para SHAP: {len(shap_values)}")
print(f"Shape de sample_X_shap_np: {sample_X_shap_np.shape}")
for i_sv, sv_arr in enumerate(shap_values):
    print(f"Shape de shap_values[{i_sv}]: {sv_arr.shape}")
    # Verificação crucial:
    if sv_arr.shape[1] != sample_X_shap_np.shape[1]:
        print(f"ALERTA: Disparidade de features para classe {i_sv}! SHAP values features: {sv_arr.shape[1]}, Data features: {sample_X_shap_np.shape[1]}")
        # Isso não deveria acontecer com TreeExplainer e dados consistentes.

# Plot SHAP Summary (Importância global das features)
plt.figure(figsize=(10,8))
# Passar sample_X_shap_df para que SHAP possa usar nomes de colunas se quiser, ou sample_X_shap_np com feature_names
shap.summary_plot(shap_values[0], sample_X_shap_df, feature_names=feature_names, show=False)
plt.title(f"SHAP Summary Plot para K-Means (Cluster {class_names[0]})")
plt.tight_layout()
shap_summary_path = os.path.join(explainability_dir, f"kmeans_shap_summary_cluster0.png")
plt.savefig(shap_summary_path)
plt.close()
print(f"Plot SHAP Summary (Cluster 0) salvo em: {shap_summary_path}")

for i, class_name_str in enumerate(class_names):
    cluster_id_for_filename = class_name_str.split()[-1]
    try:
        plt.figure(figsize=(10, max(8, len(feature_names)//2)))
        shap.summary_plot(shap_values[i], sample_X_shap_df, feature_names=feature_names, show=False, plot_type="bar")
        plt.title(f"SHAP Feature Importance para K-Means (Cluster {cluster_id_for_filename})")
        plt.tight_layout()
        shap_summary_bar_path = os.path.join(explainability_dir, f"kmeans_shap_summary_bar_cluster{cluster_id_for_filename}.png")
        plt.savefig(shap_summary_bar_path)
        plt.close()
        print(f"  Plot SHAP Summary Bar (Cluster {cluster_id_for_filename}) salvo em: {shap_summary_bar_path}")

        plt.figure(figsize=(12,8))
        shap.summary_plot(shap_values[i], sample_X_shap_df, feature_names=feature_names, show=False, plot_type="dot")
        plt.title(f"SHAP Beeswarm Plot para K-Means (Cluster {cluster_id_for_filename})")
        plt.tight_layout()
        shap_beeswarm_path = os.path.join(explainability_dir, f"kmeans_shap_beeswarm_cluster{cluster_id_for_filename}.png")
        plt.savefig(shap_beeswarm_path)
        plt.close()
        print(f"  Plot SHAP Beeswarm (Cluster {cluster_id_for_filename}) salvo em: {shap_beeswarm_path}")

    except AssertionError as ae:
        print(f"    Erro de Assertiva ao gerar plots SHAP para Cluster {cluster_id_for_filename}: {ae}. Verifique as dimensões.")
    except Exception as e:
        print(f"    Erro geral ao gerar plots SHAP para Cluster {cluster_id_for_filename}: {e}")

if hasattr(surrogate_model, 'feature_importances_'):
    importances = surrogate_model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    top_n_features = 5

    for i in range(min(top_n_features, len(feature_names))):
        feature_idx = sorted_indices[i]
        feature_name_dep = feature_names[feature_idx]
        cluster_id_for_filename_dep = class_names[0].split()[-1]
        try:
            plt.figure()
            shap.dependence_plot(feature_name_dep, shap_values[0], sample_X_shap_df, feature_names=feature_names, interaction_index="auto", show=False)
            plt.title(f"SHAP Dependence Plot: {feature_name_dep} (para Cluster {cluster_id_for_filename_dep})")
            plt.tight_layout()
            shap_dependence_path = os.path.join(explainability_dir, f"kmeans_shap_dependence_{feature_name_dep}_cluster{cluster_id_for_filename_dep}.png")
            plt.savefig(shap_dependence_path)
            plt.close()
            print(f"  Plot SHAP Dependence ({feature_name_dep}, Cluster {cluster_id_for_filename_dep}) salvo em: {shap_dependence_path}")
        except AssertionError as ae:
            print(f"    Erro de Assertiva ao gerar SHAP Dependence Plot para {feature_name_dep}: {ae}. Verifique as dimensões.")
        except Exception as e:
            print(f"    Erro geral ao gerar SHAP Dependence Plot para {feature_name_dep}: {e}")

print("\nAnálise de explicabilidade LIME e SHAP para K-Means concluída.")

