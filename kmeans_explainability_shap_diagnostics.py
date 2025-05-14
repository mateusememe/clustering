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
num_classes = len(class_names)

print("\n--- Treinando Classificador Substituto para K-Means ---")
X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
surrogate_model = RandomForestClassifier(n_estimators=100, random_state=42)
surrogate_model.fit(X_train_df, y_train)
accuracy = surrogate_model.score(X_test_df, y_test)
print(f"Acurácia do modelo substituto (RandomForest) nos dados de teste: {accuracy:.4f}")

model_path = os.path.join(explainability_dir, "kmeans_surrogate_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(surrogate_model, f)
print(f"Modelo substituto salvo em {model_path}")

# --- LIME para K-Means ---
# (Código LIME permanece o mesmo, omitido para brevidade, mas será incluído no arquivo final)
print("\n--- Aplicando LIME para K-Means (código omitido nesta visualização, mas presente no script) ---")
# ... (código LIME completo aqui)

# --- SHAP para K-Means ---
print("\n--- Aplicando SHAP para K-Means ---")

# Usar uma amostra menor para SHAP para acelerar, especialmente com KernelExplainer
sample_X_shap_df = shap.sample(X_test_df, 50) if len(X_test_df) > 50 else X_test_df
sample_X_shap_np = sample_X_shap_df.values

print(f"Tentando com TreeExplainer...")
try:
    tree_explainer = shap.TreeExplainer(surrogate_model)
    shap_values_tree = tree_explainer.shap_values(sample_X_shap_np)
    print(f"  Tipo de shap_values_tree: {type(shap_values_tree)}")
    if isinstance(shap_values_tree, list):
        print(f"  shap_values_tree é uma lista de {len(shap_values_tree)} arrays.")
        for i_sv, sv_arr in enumerate(shap_values_tree):
            print(f"    Shape de shap_values_tree[{i_sv}]: {sv_arr.shape}")
    else:
        print(f"  Shape de shap_values_tree (não é lista): {shap_values_tree.shape}")
        # Se for um array 3D (n_samples, n_features, n_classes) ou (n_samples, n_classes, n_features)
        # ou (n_classes, n_samples, n_features) - o padrão do SHAP para multi-classe é lista de (n_samples, n_features)
        # Se for (n_samples, n_features, n_classes), precisamos transpor/reorganizar para o formato de lista
        if shap_values_tree.ndim == 3 and shap_values_tree.shape[0] == sample_X_shap_np.shape[0] and shap_values_tree.shape[1] == sample_X_shap_np.shape[1] and shap_values_tree.shape[2] == num_classes:
            print("  Formato detectado: (n_samples, n_features, n_classes). Reorganizando para lista de arrays (n_samples, n_features) por classe.")
            temp_list = []
            for i_class in range(num_classes):
                temp_list.append(shap_values_tree[:, :, i_class])
            shap_values_tree = temp_list
            print(f"  Reorganizado. Agora é uma lista de {len(shap_values_tree)} arrays.")
        elif shap_values_tree.ndim == 2 and num_classes == 2: # Caso binário especial onde pode retornar para a classe positiva
             print("  Caso binário detectado, TreeExplainer pode retornar SHAP values apenas para a classe positiva.")
             # Precisamos criar a lista com SHAP values para ambas as classes (-shap_values, shap_values)
             shap_values_class_0 = -shap_values_tree
             shap_values_class_1 = shap_values_tree
             shap_values_tree = [shap_values_class_0, shap_values_class_1]
             print(f"  Reorganizado para caso binário. Agora é uma lista de {len(shap_values_tree)} arrays.")
        else:
            print("  Formato de shap_values_tree não reconhecido para conversão automática em lista.")

    shap_values_to_plot = shap_values_tree
    explainer_type = "TreeExplainer"

except Exception as e_tree:
    print(f"  Erro com TreeExplainer: {e_tree}")
    print(f"  Tentando com KernelExplainer como fallback...")
    # KernelExplainer é mais lento, usar um background menor
    background_sample_kernel = shap.sample(X_train_df, 30) 
    kernel_explainer = shap.KernelExplainer(surrogate_model.predict_proba, background_sample_kernel)
    shap_values_kernel = kernel_explainer.shap_values(sample_X_shap_np)
    print(f"  Tipo de shap_values_kernel: {type(shap_values_kernel)}")
    if isinstance(shap_values_kernel, list):
        print(f"  shap_values_kernel é uma lista de {len(shap_values_kernel)} arrays.")
        for i_sv, sv_arr in enumerate(shap_values_kernel):
            print(f"    Shape de shap_values_kernel[{i_sv}]: {sv_arr.shape}")
    else:
        print(f"  Shape de shap_values_kernel (não é lista): {shap_values_kernel.shape}")
    shap_values_to_plot = shap_values_kernel
    explainer_type = "KernelExplainer"

if not isinstance(shap_values_to_plot, list) or not all(isinstance(arr, np.ndarray) for arr in shap_values_to_plot):
    print("ERRO: shap_values_to_plot não está no formato esperado (lista de arrays NumPy). Abortando plots SHAP.")
    exit()
if len(shap_values_to_plot) != num_classes:
    print(f"ERRO: Número de arrays em shap_values_to_plot ({len(shap_values_to_plot)}) não corresponde ao número de classes ({num_classes}). Abortando.")
    exit()

print(f"Usando {explainer_type} para plots SHAP.")

# Plot SHAP Summary (Importância global das features)
# Verifique se sample_X_shap_df está correto para os plots
if sample_X_shap_df.shape[0] != shap_values_to_plot[0].shape[0]:
    print(f"ERRO: Disparidade de amostras entre sample_X_shap_df ({sample_X_shap_df.shape[0]}) e shap_values ({shap_values_to_plot[0].shape[0]}).")
    exit()

for i, class_name_str in enumerate(class_names):
    cluster_id_for_filename = class_name_str.split()[-1]
    try:
        plt.figure(figsize=(10, max(8, len(feature_names)//2)))
        shap.summary_plot(shap_values_to_plot[i], sample_X_shap_df, feature_names=feature_names, show=False, plot_type="bar")
        plt.title(f"SHAP Feature Importance ({explainer_type}) para K-Means (Cluster {cluster_id_for_filename})")
        plt.tight_layout()
        shap_summary_bar_path = os.path.join(explainability_dir, f"kmeans_shap_summary_bar_cluster{cluster_id_for_filename}_{explainer_type}.png")
        plt.savefig(shap_summary_bar_path)
        plt.close()
        print(f"  Plot SHAP Summary Bar (Cluster {cluster_id_for_filename}) salvo em: {shap_summary_bar_path}")

        plt.figure(figsize=(12,8))
        shap.summary_plot(shap_values_to_plot[i], sample_X_shap_df, feature_names=feature_names, show=False, plot_type="dot")
        plt.title(f"SHAP Beeswarm Plot ({explainer_type}) para K-Means (Cluster {cluster_id_for_filename})")
        plt.tight_layout()
        shap_beeswarm_path = os.path.join(explainability_dir, f"kmeans_shap_beeswarm_cluster{cluster_id_for_filename}_{explainer_type}.png")
        plt.savefig(shap_beeswarm_path)
        plt.close()
        print(f"  Plot SHAP Beeswarm (Cluster {cluster_id_for_filename}) salvo em: {shap_beeswarm_path}")

    except AssertionError as ae:
        print(f"    Erro de Assertiva ao gerar plots SHAP para Cluster {cluster_id_for_filename}: {ae}. Verifique as dimensões.")
    except Exception as e:
        print(f"    Erro geral ao gerar plots SHAP para Cluster {cluster_id_for_filename}: {e}")

if hasattr(surrogate_model, 'feature_importances_') and isinstance(shap_values_to_plot, list) and len(shap_values_to_plot) > 0:
    importances = surrogate_model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    top_n_features = 3 # Reduzido para menos plots de dependência

    for i_feat in range(min(top_n_features, len(feature_names))):
        feature_idx = sorted_indices[i_feat]
        feature_name_dep = feature_names[feature_idx]
        # Plotar para a primeira classe como exemplo
        class_idx_for_dep_plot = 0 
        cluster_id_for_filename_dep = class_names[class_idx_for_dep_plot].split()[-1]
        try:
            plt.figure()
            shap.dependence_plot(feature_name_dep, shap_values_to_plot[class_idx_for_dep_plot], sample_X_shap_df, 
                                 feature_names=feature_names, interaction_index="auto", show=False)
            plt.title(f"SHAP Dependence: {feature_name_dep} ({explainer_type}, Cluster {cluster_id_for_filename_dep})")
            plt.tight_layout()
            shap_dependence_path = os.path.join(explainability_dir, f"kmeans_shap_dependence_{feature_name_dep}_cluster{cluster_id_for_filename_dep}_{explainer_type}.png")
            plt.savefig(shap_dependence_path)
            plt.close()
            print(f"  Plot SHAP Dependence ({feature_name_dep}, Cluster {cluster_id_for_filename_dep}) salvo em: {shap_dependence_path}")
        except AssertionError as ae:
            print(f"    Erro de Assertiva ao gerar SHAP Dependence Plot para {feature_name_dep}: {ae}.")
        except Exception as e:
            print(f"    Erro geral ao gerar SHAP Dependence Plot para {feature_name_dep}: {e}")

print("\nAnálise de explicabilidade LIME e SHAP para K-Means concluída.")

# Recriar a parte LIME que foi omitida para o teste
print("\n--- Re-executando LIME para K-Means (para garantir que está completo no script final) ---")
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
        # print(f"    Explicação LIME (HTML) salva em: {lime_html_path}") # Já impresso antes
        fig = explanation_lime.as_pyplot_figure(label=explanation_lime.available_labels()[0])
        plt.title(f"LIME para Instância (test_idx {instance_idx_in_test}) - Prev. Cluster {explanation_lime.available_labels()[0]} (Real: {true_cluster})")
        plt.tight_layout()
        lime_plot_path = os.path.join(explainability_dir, f"kmeans_lime_instance_testidx{instance_idx_in_test}_cluster{true_cluster}.png")
        plt.savefig(lime_plot_path)
        plt.close(fig)
        # print(f"    Plot LIME salvo em: {lime_plot_path}") # Já impresso antes
    except Exception as e:
        print(f"    Erro ao gerar explicação LIME para instância com índice {instance_idx_in_test} em X_test_df: {e}")
print("--- Fim da re-execução LIME ---")

