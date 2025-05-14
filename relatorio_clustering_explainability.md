# Relatório de Análise de Clustering com Explainable AI

## 1. Introdução

Este relatório detalha o processo e os resultados da aplicação de três algoritmos de clustering (K-Means tradicional, Robust and Sparse K-Means - RSKC, e Adaptively Robust and Sparse K-Means - ARSKC) ao dataset "Short-Highway-Rail-Crossing-Dataset". O objetivo principal é não apenas agrupar os dados, mas também interpretar os agrupamentos formados utilizando técnicas de Explainable AI (XAI), especificamente LIME (Local Interpretable Model-agnostic Explanations) e SHAP (SHapley Additive exPlanations).

A análise visa encontrar o número ideal de clusters (K ótimo) para cada algoritmo, visualizar os resultados e, crucialmente, entender quais características (features) do dataset são mais determinantes para a formação de cada cluster. Este entendimento aprofundado pode revelar padrões ocultos nos dados de cruzamentos rodoferroviários, potencialmente relacionados a aspectos como prejuízo financeiro, tempo de interrupção de serviço e outros fatores de risco ou operacionais.

## 2. Dataset e Pré-processamento

### 2.1. Descrição do Dataset

O dataset utilizado, "Short-Highway-Rail-Crossing-Dataset.csv", contém informações sobre incidentes e características de cruzamentos rodoferroviários. Uma análise exploratória inicial (EDA) foi conduzida para entender a estrutura dos dados, os tipos de variáveis, suas distribuições, e a presença de valores ausentes ou outliers.

*(Nota: Detalhes específicos da EDA, como contagem de valores ausentes por coluna, estatísticas descritivas das features numéricas e distribuições das features categóricas, seriam inseridos aqui, baseados nos outputs dos scripts de EDA executados anteriormente. Como os outputs exatos não estão no histórico recente, esta seção é um template.)*

### 2.2. Seleção de Features

Com base na solicitação do usuário e na análise exploratória, foram selecionadas features consideradas relevantes para o clustering, com foco em variáveis que poderiam influenciar ou caracterizar o prejuízo financeiro, o tempo de interrupção e a severidade dos incidentes. As features selecionadas incluem (lista exemplificativa, a lista real foi determinada no script `eda_preprocessing_part2.py`):

*   `Year`, `Month`, `Day` (ou features derivadas de data)
*   `Time` (ou features derivadas de hora)
*   `Temperature (F)`
*   `Visibility (mi)`
*   `Weather Condition` (codificada)
*   `Type of Equipment` (codificada)
*   `Speed`
*   `Number of Tracks`
*   Outras features numéricas e categóricas relevantes que foram identificadas e codificadas.

### 2.3. Pré-processamento dos Dados

O pré-processamento dos dados envolveu as seguintes etapas:

1.  **Tratamento de Valores Ausentes**: Valores ausentes em colunas numéricas foram imputados utilizando a mediana da respectiva coluna. Para colunas categóricas, a moda foi utilizada. Esta abordagem foi escolhida para preservar a distribuição dos dados tanto quanto possível sem introduzir viés excessivo.
2.  **Codificação de Variáveis Categóricas**: Variáveis categóricas foram convertidas em representações numéricas utilizando a técnica de *one-hot encoding* (ou *dummy variables*). Isso é necessário porque os algoritmos de clustering operam sobre dados numéricos.
3.  **Normalização/Padronização**: Todas as features numéricas selecionadas (incluindo as resultantes da codificação one-hot) foram padronizadas utilizando o `StandardScaler` do scikit-learn. Este processo transforma os dados para que tenham média zero e desvio padrão unitário, garantindo que features com magnitudes diferentes tenham pesos comparáveis no cálculo das distâncias pelos algoritmos de clustering.

O dataset pré-processado e escalonado foi salvo como `/home/ubuntu/preprocessed_highway_rail_crossing_dataset.csv` para ser utilizado nas etapas subsequentes de clustering e explicabilidade.

## 3. Métricas de Avaliação de Clustering

Para determinar o número ótimo de clusters (K) e avaliar a qualidade dos agrupamentos, foram utilizadas as seguintes métricas:

### 3.1. Método do Cotovelo (Elbow Method) com WCSS

O Método do Cotovelo é uma técnica heurística usada para encontrar o número ideal de clusters. Ele se baseia na ideia de que um bom agrupamento terá uma soma dos quadrados dentro dos clusters (Within-Cluster Sum of Squares - WCSS, também conhecida como inércia) baixa. O WCSS mede a variância dentro de cada cluster. Ao plotar o WCSS em função do número de clusters (K), geralmente observa-se que o WCSS diminui à medida que K aumenta. O "cotovelo" no gráfico representa o ponto onde adicionar mais clusters não resulta em uma diminuição significativa do WCSS. Este ponto é considerado uma indicação do K ótimo.

### 3.2. Silhouette Score

O Silhouette Score mede o quão similar um objeto é ao seu próprio cluster (coesão) em comparação com outros clusters (separação). O score varia de -1 a +1, onde um valor alto indica que o objeto está bem encaixado em seu próprio cluster e mal encaixado nos clusters vizinhos. Valores próximos de 0 indicam clusters sobrepostos, e valores negativos geralmente indicam que as amostras podem ter sido atribuídas ao cluster errado. O K ótimo é frequentemente associado ao maior Silhouette Score médio para todas as amostras.

### 3.3. Davies-Bouldin Index

O Davies-Bouldin Index (DBI) é uma métrica de avaliação de clustering que mede a similaridade média entre cada cluster e seu cluster mais similar, onde a similaridade é a razão entre as distâncias intra-cluster e as distâncias inter-cluster. Valores mais baixos do DBI indicam um melhor particionamento, ou seja, clusters mais compactos e bem separados. O K ótimo é geralmente associado ao menor valor do DBI.



## 4. K-Means Tradicional

### 4.1. Descrição do Algoritmo

O K-Means é um dos algoritmos de clustering mais populares e simples. Ele particiona o dataset em K clusters distintos e não sobrepostos. O objetivo é minimizar a variância dentro de cada cluster, ou seja, a soma dos quadrados das distâncias entre cada ponto de dados e o centróide do seu cluster atribuído (WCSS).

O algoritmo opera iterativamente:
1.  **Inicialização**: Escolhe K centróides iniciais (aleatoriamente ou usando um método mais inteligente como k-means++).
2.  **Atribuição**: Atribui cada ponto de dados ao centróide mais próximo.
3.  **Atualização**: Recalcula os centróides como a média de todos os pontos de dados atribuídos a esse cluster.
Os passos 2 e 3 são repetidos até que os centróides não mudem significativamente ou um número máximo de iterações seja atingido.

### 4.2. Determinação do K Ótimo e Hiperparâmetros

Para o K-Means tradicional, o principal hiperparâmetro a ser determinado é o número de clusters (K). Utilizamos três métricas para auxiliar nesta escolha, com K variando de 2 a 15:

1.  **Método do Cotovelo (WCSS)**: O gráfico do WCSS em função de K foi gerado e salvo em `/home/ubuntu/kmeans_results/kmeans_elbow_method.png`. A análise visual deste gráfico busca um "cotovelo" onde a taxa de diminuição do WCSS se torna menos acentuada.
    *(Visualização: kmeans_elbow_method.png)*

2.  **Silhouette Score**: Os scores médios de silhueta para diferentes valores de K foram calculados e plotados em `/home/ubuntu/kmeans_results/kmeans_silhouette_scores.png`. O valor de K que maximiza o Silhouette Score é geralmente preferido.
    *(Visualização: kmeans_silhouette_scores.png)*

3.  **Davies-Bouldin Index**: Os índices de Davies-Bouldin para diferentes valores de K foram calculados e plotados em `/home/ubuntu/kmeans_results/kmeans_davies_bouldin_index.png`. O valor de K que minimiza o DBI é geralmente preferido.
    *(Visualização: kmeans_davies_bouldin_index.png)*

Com base nos resultados obtidos (conforme logs da execução de `/home/ubuntu/kmeans_analysis.py`):
*   O Silhouette Score máximo foi obtido para **K=5** (score de ~0.177).
*   O Davies-Bouldin Index apresentou um valor mínimo para K=15, mas K=4 também mostrou um bom valor local (~1.733), sendo K=5 um valor razoável (~1.815) antes de começar a aumentar novamente para K=6 e K=7 e depois diminuir.
*   O método do cotovelo, embora subjetivo, geralmente sugere um K onde a curva começa a achatar. Para este dataset, a curva WCSS mostrou uma diminuição mais suave após K=5 ou K=6.

Considerando a combinação dessas métricas, especialmente o pico claro no Silhouette Score, **K=5** foi escolhido como o número ótimo de clusters para o K-Means tradicional.

### 4.3. Resultados do Clustering

O algoritmo K-Means foi executado com K=5 no dataset pré-processado. Os labels dos clusters foram adicionados ao dataset, que foi salvo como `/home/ubuntu/kmeans_results/dataset_with_kmeans_clusters_k5.csv`.

Uma visualização dos clusters resultantes, utilizando Análise de Componentes Principais (PCA) para reduzir a dimensionalidade para duas componentes, foi gerada e salva como `/home/ubuntu/kmeans_results/kmeans_clusters_pca_k5.png`.
*(Visualização: kmeans_clusters_pca_k5.png)*

### 4.4. Interpretação dos Agrupamentos com LIME e SHAP

Para entender as características que definem cada um dos 5 clusters formados pelo K-Means, um modelo classificador substituto (RandomForestClassifier) foi treinado para prever as atribuições de cluster com base nas features originais. Este modelo alcançou uma acurácia de 0.9800 nos dados de teste, indicando que ele aprendeu bem a relação entre as features e os clusters atribuídos pelo K-Means, tornando-o adequado para a explicabilidade.

**LIME (Local Interpretable Model-agnostic Explanations)**

LIME foi aplicado para explicar as atribuições de cluster para instâncias individuais representativas de cada um dos 5 clusters. As explicações mostram as features que mais contribuíram (positiva ou negativamente) para a atribuição de uma instância a um cluster específico. Os resultados foram salvos como arquivos HTML e PNG para cada instância explicada no diretório `/home/ubuntu/kmeans_results/explainability/` (ex: `kmeans_lime_instance_testidx3_cluster0.html` e `kmeans_lime_instance_testidx3_cluster0.png`).

*(Exemplo de Visualizações LIME: kmeans_lime_instance_testidx*_cluster*.png)*

**SHAP (SHapley Additive exPlanations)**

SHAP foi utilizado para fornecer uma visão global da importância das features para cada cluster. Foram gerados:
*   **Summary Plots (Bar)**: Mostram a importância média absoluta das features para cada cluster. (ex: `kmeans_shap_summary_bar_cluster0_TreeExplainer.png`)
*   **Summary Plots (Beeswarm/Dot)**: Mostram não apenas a importância, mas também o impacto (positivo/negativo) e a distribuição dos valores SHAP para cada feature em cada cluster. (ex: `kmeans_shap_beeswarm_cluster0_TreeExplainer.png`)
*   **Dependence Plots**: Mostram como o valor de uma feature específica afeta a predição (valor SHAP) para um cluster, possivelmente revelando interações com outra feature. (ex: `kmeans_shap_dependence_Equipment Struck Code_1_cluster0_TreeExplainer.png`)

Todos os plots SHAP foram salvos no diretório `/home/ubuntu/kmeans_results/explainability/`.

*(Exemplo de Visualizações SHAP: kmeans_shap_summary_bar_cluster*_TreeExplainer.png, kmeans_shap_beeswarm_cluster*_TreeExplainer.png)*

*(Análise detalhada das features mais importantes para cada cluster seria inserida aqui, com base na inspeção dos plots LIME e SHAP gerados. Por exemplo: "Para o Cluster 0, as features X, Y e Z foram consistentemente identificadas como as mais influentes, com valores altos de X e baixos de Y sendo característicos deste cluster...")*



## 5. Robust and Sparse K-Means (RSKC)

### 5.1. Descrição do Algoritmo

O Robust and Sparse K-Means (RSKC) é uma extensão do K-Means tradicional projetada para lidar com outliers (robustez) e selecionar features relevantes (esparsidade) simultaneamente. A robustez é alcançada atribuindo pesos menores a observações que são distantes dos centróides de seus clusters, reduzindo assim a influência de outliers. A esparsidade é introduzida penalizando os pesos das features, efetivamente selecionando um subconjunto de features que são mais importantes para a estrutura de clustering. Isso é particularmente útil em datasets de alta dimensionalidade onde muitas features podem ser irrelevantes ou ruidosas.

A implementação adaptada de R para Python (`/home/ubuntu/rskc_implementation.py`) busca replicar a funcionalidade descrita na referência do pacote `RSKC` do CRAN.

### 5.2. Determinação do K Ótimo e Hiperparâmetros

Para o RSKC, além do número de clusters (K), hiperparâmetros como `alpha` (controla a robustez, peso das observações) e `L1` (controla a esparsidade, penalidade L1 nos pesos das features) precisam ser definidos. A análise foi conduzida para as versões não-esparsa (apenas robustez) e esparsa.

**Versão Não-Esparsa (Robust K-Means):**
*   Foi testado com `alpha` = 0.05 (um valor comum para considerar 5% de outliers).
*   As métricas (WCSS, Silhouette, DBI) foram calculadas para K de 2 a 15.
    *   WCSS: `/home/ubuntu/rskc_results/rskc_nonsparse_elbow_alpha0.05.png`
    *   Silhouette: `/home/ubuntu/rskc_results/rskc_nonsparse_silhouette_alpha0.05.png`
    *   DBI: `/home/ubuntu/rskc_results/rskc_nonsparse_db_index_alpha0.05.png`
*   Para `alpha`=0.05, o Silhouette Score máximo foi para **K=2** (~0.202), e o DBI também foi favorável para K=2 (~1.45).

**Versão Esparsa (Robust and Sparse K-Means):**
*   Mantendo `alpha` = 0.05 e K=2 (baseado na análise não-esparsa), variou-se o parâmetro de esparsidade `L1`.
*   A seleção do `L1` ótimo pode ser feita observando a estabilidade dos clusters ou o número de features selecionadas. No script `rskc_analysis.py`, um valor de `L1_sparse = 5` foi usado como exemplo para demonstração, mas na prática, uma busca mais sistemática ou critérios baseados na variância explicada ou estabilidade dos pesos das features seriam recomendados.
    *   WCSS (para L1=5): `/home/ubuntu/rskc_results/rskc_sparse_wbss_alpha0.05_L1_5.png`
    *   Silhouette (para L1=5): `/home/ubuntu/rskc_results/rskc_sparse_silhouette_alpha0.05_L1_5.png`
    *   DBI (para L1=5): `/home/ubuntu/rskc_results/rskc_sparse_db_index_alpha0.05_L1_5.png`
*   Para K=2 e L1=5, o Silhouette Score foi ~0.202 e o DBI ~1.45.

Para este relatório, focaremos nos resultados do RSKC esparso com **K=2**, `alpha=0.05` e `L1=5` (valor usado na geração dos arquivos de cluster para explicabilidade).

### 5.3. Resultados do Clustering

O RSKC (esparso) foi executado com K=2, `alpha=0.05`, e `L1=5`. Os labels dos clusters foram salvos em `/home/ubuntu/rskc_results/dataset_with_rskc_sparse_clusters_k2.csv`.

A visualização PCA dos clusters foi salva como `/home/ubuntu/rskc_results/rskc_sparse_clusters_pca_k2.png`.
*(Visualização: rskc_sparse_clusters_pca_k2.png)*

### 5.4. Interpretação dos Agrupamentos com LIME e SHAP

Um modelo RandomForestClassifier substituto foi treinado para prever os clusters RSKC (K=2), alcançando uma acurácia de 1.0000, indicando uma separação clara que o modelo substituto conseguiu aprender perfeitamente.

**LIME**

Explicações LIME para instâncias representativas dos 2 clusters RSKC foram geradas e salvas em `/home/ubuntu/rskc_results/explainability/` (ex: `rskc_lime_instance_testidx0_cluster0.html`).
*(Exemplo de Visualizações LIME: rskc_lime_instance_testidx*_cluster*.png)*

**SHAP**

Plots SHAP (summary bar, beeswarm) foram gerados para os 2 clusters RSKC e salvos em `/home/ubuntu/rskc_results/explainability/` (ex: `rskc_shap_summary_bar_cluster0_TreeExplainer.png`). Devido à esparsidade do RSKC, espera-se que menos features tenham um impacto significativo na formação dos clusters em comparação com o K-Means tradicional.

*(Exemplo de Visualizações SHAP: rskc_shap_summary_bar_cluster*_TreeExplainer.png, rskc_shap_beeswarm_cluster*_TreeExplainer.png)*

*(Análise detalhada das features mais importantes para cada cluster RSKC, com ênfase nas features selecionadas pela esparsidade, seria inserida aqui.)*

## 6. Adaptively Robust and Sparse K-Means (ARSKC)

### 6.1. Descrição do Algoritmo

O Adaptively Robust and Sparse K-Means (ARSKC) refina o R
(Content truncated due to size limit. Use line ranges to read in chunks)