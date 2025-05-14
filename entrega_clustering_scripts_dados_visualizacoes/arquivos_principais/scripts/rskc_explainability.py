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

# Imprimir versão do SHAP
print(f"Versão da biblioteca SHAP: {shap.__version__}")

# Caminhos
preprocessed_dataset_path = "/home/ubuntu/preprocessed_highway_rail_crossing_dataset.csv"
rskc_results_dir = "/home/ubuntu/rskc_results"
explainability_dir = os.path.join(rskc_results_dir, "explainability")

# Assumindo K=2 para RSKC (sparse) conforme análise anterior
K_OPTIMAL_RSKC = 2 
clustered_data_path = os.path.join(rskc_results_dir, f"dataset_with_rskc_sparse_clusters_k{K_OPTIMAL_RSKC}.csv")

# Certificar que o diretório de resultados existe
os.makedirs(explainability_dir, exist_ok=True)

print(f"Carregando dataset pré-processado de {preprocessed_dataset_path}")
try:
    df_scaled = pd.read_csv(preprocessed_dataset_path)
    print(f"Dataset pré-processado carregado. Dimensões: {df_scaled.shape}")
except Exception as e:
    print(f"Erro ao carregar o dataset pré-processado: {e}")
    exit()

print(f"Carregando dataset com clusters RSKC de {clustered_data_path}")
try:
    df_clustered_rskc = pd.read_csv(clustered_data_path)
    if "Cluster_RSKC_Sparse" not in df_clustered_rskc.columns:
        print(f"Erro: Coluna 'Cluster_RSKC_Sparse' não encontrada no arquivo {clustered_data_path}.")
        # Tentar com o nome não esparso se o esparso não existir com K=2
        if K_OPTIMAL_RSKC == 2 and os.path.exists(os.path.join(rskc_results_dir, f"dataset_with_rskc_nonsparse_clusters_k{K_OPTIMAL_RSKC}.csv")):
            clustered_data_path = os.path.join(rskc_results_dir, f"dataset_with_rskc_nonsparse_clusters_k{K_OPTIMAL_RSKC}.csv")
            print(f"Tentando com {clustered_data_path}")
            df_clustered_rskc = pd.read_csv(clustered_data_path)
            if "Cluster_RSKC_NonSparse" not in df_clustered_rskc.columns:
                print(f"Erro: Coluna 'Cluster_RSKC_NonSparse' também não encontrada.")
                exit()
            rskc_labels_column = "Cluster_RSKC_NonSparse"
        else:
            exit()
    else:
        rskc_labels_column = "Cluster_RSKC_Sparse"

    rskc_labels = df_clustered_rskc[rskc_labels_column].values
    if df_scaled.shape[0] != len(rskc_labels):
        print("Erro: Disparidade no número de amostras entre dados escalados e labels do RSKC.")
        exit()
    print(f"Labels RSKC ({rskc_labels_column}) carregados. Número de clusters: {len(np.unique(rskc_labels))}")
except Exception as e:
    print(f"Erro ao carregar o dataset com clusters RSKC: {e}")
    exit()

# Preparar dados para o classificador substituto
X = df_scaled.copy()
y = rskc_labels
feature_names = X.columns.tolist()
class_names = [f"Cluster {i}" for i in sorted(np.unique(y))]
num_classes = len(class_names)

print("\n--- Treinando Classificador Substituto para RSKC ---")
X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
surrogate_model = RandomForestClassifier(n_estimators=100, random_state=42)
surrogate_model.fit(X_train_df, y_train)
accuracy = surrogate_model.score(X_test_df, y_test)
print(f"Acurácia do modelo substituto (RandomForest) para RSKC: {accuracy:.4f}")

model_path = os.path.join(explainability_dir, f"rskc_surrogate_model_k{K_OPTIMAL_RSKC}.pkl")
with open(model_path, "wb") as f:
    pickle.dump(surrogate_model, f)
print(f"Modelo substituto RSKC salvo em {model_path}")

# --- LIME para RSKC ---
print("\n--- Aplicando LIME para RSKC ---")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_df.values,
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
    print(f"  Explicando instância RSKC (test_idx {instance_idx_in_test}) - Cluster Real: {true_cluster}")
    try:
        explanation_lime = lime_explainer.explain_instance(
            data_row=instance_series.values,
            predict_fn=surrogate_model.predict_proba,
            num_features=10,
            top_labels=1
        )
        lime_html_path = os.path.join(explainability_dir, f"rskc_lime_instance_testidx{instance_idx_in_test}_cluster{true_cluster}.html")
        explanation_lime.save_to_file(lime_html_path)
        print(f"    Explicação LIME RSKC (HTML) salva em: {lime_html_path}")
        fig = explanation_lime.as_pyplot_figure(label=explanation_lime.available_labels()[0])
        plt.title(f"LIME RSKC (test_idx {instance_idx_in_test}) - Prev. Cluster {explanation_lime.available_labels()[0]} (Real: {true_cluster})")
        plt.tight_layout()
        lime_plot_path = os.path.join(explainability_dir, f"rskc_lime_instance_testidx{instance_idx_in_test}_cluster{true_cluster}.png")
        plt.savefig(lime_plot_path)
        plt.close(fig)
        print(f"    Plot LIME RSKC salvo em: {lime_plot_path}")
    except Exception as e:
        print(f"    Erro ao gerar explicação LIME RSKC para instância {instance_idx_in_test}: {e}")

# --- SHAP para RSKC ---
print("\n--- Aplicando SHAP para RSKC ---")
sample_X_shap_df = shap.sample(X_test_df, 50) if len(X_test_df) > 50 else X_test_df
sample_X_shap_np = sample_X_shap_df.values
explainer_type = ""

try:
    tree_explainer = shap.TreeExplainer(surrogate_model)
    shap_values_tree = tree_explainer.shap_values(sample_X_shap_np)
    if isinstance(shap_values_tree, list):
        shap_values_to_plot = shap_values_tree
    elif shap_values_tree.ndim == 3 and shap_values_tree.shape[0] == sample_X_shap_np.shape[0] and shap_values_tree.shape[1] == sample_X_shap_np.shape[1] and shap_values_tree.shape[2] == num_classes:
        temp_list = [shap_values_tree[:, :, i_class] for i_class in range(num_classes)]
        shap_values_to_plot = temp_list
    elif shap_values_tree.ndim == 2 and num_classes == 2:
        shap_values_to_plot = [-shap_values_tree, shap_values_tree]
    else:
        raise ValueError("Formato de SHAP values do TreeExplainer não compatível.")
    explainer_type = "TreeExplainer"
except Exception as e_tree:
    print(f"  Erro com TreeExplainer para RSKC: {e_tree}. Tentando KernelExplainer.")
    background_sample_kernel = shap.sample(X_train_df, 30)
    kernel_explainer = shap.KernelExplainer(surrogate_model.predict_proba, background_sample_kernel)
    shap_values_kernel = kernel_explainer.shap_values(sample_X_shap_np)
    shap_values_to_plot = shap_values_kernel
    explainer_type = "KernelExplainer"

if not isinstance(shap_values_to_plot, list) or len(shap_values_to_plot) != num_classes:
    print("ERRO: SHAP values para RSKC não estão no formato esperado. Abortando plots SHAP.")
    exit()

print(f"Usando {explainer_type} para plots SHAP (RSKC).")

for i, class_name_str in enumerate(class_names):
    cluster_id_for_filename = class_name_str.split()[-1]
    try:
        plt.figure(figsize=(10, max(8, len(feature_names)//2)))
        shap.summary_plot(shap_values_to_plot[i], sample_X_shap_df, feature_names=feature_names, show=False, plot_type="bar")
        plt.title(f"SHAP Feature Importance ({explainer_type}) para RSKC (Cluster {cluster_id_for_filename})")
        plt.tight_layout()
        shap_summary_bar_path = os.path.join(explainability_dir, f"rskc_shap_summary_bar_cluster{cluster_id_for_filename}_{explainer_type}.png")
        plt.savefig(shap_summary_bar_path)
        plt.close()
        print(f"  Plot SHAP Summary Bar RSKC (Cluster {cluster_id_for_filename}) salvo em: {shap_summary_bar_path}")

        plt.figure(figsize=(12,8))
        shap.summary_plot(shap_values_to_plot[i], sample_X_shap_df, feature_names=feature_names, show=False, plot_type="dot")
        plt.title(f"SHAP Beeswarm Plot ({explainer_type}) para RSKC (Cluster {cluster_id_for_filename})")
        plt.tight_layout()
        shap_beeswarm_path = os.path.join(explainability_dir, f"rskc_shap_beeswarm_cluster{cluster_id_for_filename}_{explainer_type}.png")
        plt.savefig(shap_beeswarm_path)
        plt.close()
        print(f"  Plot SHAP Beeswarm RSKC (Cluster {cluster_id_for_filename}) salvo em: {shap_beeswarm_path}")
    except Exception as e:
        print(f"    Erro ao gerar plots SHAP RSKC para Cluster {cluster_id_for_filename}: {e}")

print("\nAnálise de explicabilidade LIME e SHAP para RSKC concluída.")

