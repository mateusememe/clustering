# Lista de Tarefas - Análise de Clustering com Explainable AI

## 1. Análise e Preparação do Dataset

- [X] Carregar o dataset "Short-Highway-Rail-Crossing-Dataset.csv".
- [X] Realizar análise exploratória dos dados (EDA) para entender a estrutura, tipos de dados, distribuição das variáveis, identificar valores ausentes e outliers.
- [X] Identificar e selecionar features relevantes para o clustering, com foco em colunas que possam indicar prejuízo financeiro, tempo de interrupção, severidade de incidentes, ou características que diferenciem os cruzamentos rodoferroviários.
- [X] Realizar pré-processamento dos dados:
    - [X] Tratar valores ausentes (se houver), justificando a estratégia adotada (remoção, imputação, etc.).
    - [X] Codificar variáveis categóricas (se necessário) para que possam ser usadas pelos algoritmos de clustering.
    - [X] Realizar a normalização/padronização das features selecionadas para que todas tenham a mesma escala.
- [X] Salvar o dataset pré-processado para uso nas etapas seguintes.

## 2. Implementação e Adaptação dos Algoritmos

- [X] Implementar o algoritmo K-Means tradicional utilizando bibliotecas Python (ex: scikit-learn).
- [X] Pesquisar a teoria e a implementação de referência do Robust and Sparse K-Means (RSKC) a partir de "https://rdrr.io/cran/RSKC/man/RSKC.html" e dos artigos fornecidos.
- [X] Adaptar ou implementar o algoritmo RSKC em Python. (Implementação funcional concluída e validada).
- [X] Pesquisar a teoria e a implementação de referência do Adaptively Robust and Sparse K-Means (ARSKC) a partir de "https://github.com/lee1995hao/ARSK" (código em R) e dos artigos fornecidos.
- [X] Traduzir e adaptar a implementação do ARSKC de R para Python. (Implementação funcional concluída e validada).

## 3. Aplicação do K-Means Tradicional

- [X] Implementar a função para determinar o número ótimo de clusters (K ótimo) utilizando o método do cotovelo (Elbow Method). Plotar o gráfico do cotovelo.
- [X] Implementar a avaliação do K ótimo utilizando o Silhouette Score. Plotar os scores para diferentes valores de K.
- [X] Implementar a avaliação do K ótimo utilizando o Davies-Bouldin Index. Plotar os índices para diferentes valores de K.
- [X] Justificar a escolha do K ótimo com base nas três métricas.
- [X] Executar o algoritmo K-Means com o K ótimo encontrado no dataset pré-processado.
- [X] Gerar visualizações dos resultados do clustering (ex: scatter plot com clusters coloridos, usando PCA para redução de dimensionalidade se necessário).
- [X] Aplicar LIME para explicar as predições de cluster para instâncias individuais selecionadas.
- [X] Aplicar SHAP para obter uma visão global da importância das features para cada cluster e para o modelo de clustering como um todo.
- [X] Salvar as visualizações e os resultados da interpretabilidade.## 4. Aplicação do Robust and Sparse K-Means (RSKC)

- [X] Se aplicável, determinar o número ótimo de clusters (K ótimo) para o RSKC usando o método do cotovelo, Silhouette Score e Davies-Bouldin Index. Justificar a escolha.
- [X] Identificar os hiperparâmetros do RSKC (ex: lambda para esparsidade/robustez). Pesquisar ou definir uma estratégia para encontrar valores adequados para esses hiperparâmetros (ex: grid search, validação cruzada se aplicável, ou com base na literatura). Explicar a abordagem.
- [X] Executar o algoritmo RSKC com os parâmetros ótimos no dataset pré-processado.
- [X] Gerar visualizações dos resultados do clustering.
- [X] Aplicar LIME para explicar as predições de cluster.
- [X] Aplicar SHAP para obter uma visão global da importância das features.
- [X] Salvar as visualizações e os resultados da interpretabilidade.

## 5. Aplicação do Adaptively Robust and Sparse K-Means (ARSKC)

- [X] Se aplicável, determinar o número ótimo de clusters (K ótimo) para o ARSKC usando o método do cotovelo, Silhouette Score e Davies-Bouldin Index. Justificar a escolha.
- [X] Identificar os hiperparâmetros do ARSKC (ex: lambda, gamma). Pesquisar ou definir uma estratégia para encontrar valores adequados. Explicar a abordagem.
- [X] Executar o algoritmo ARSKC com os parâmetros ótimos no dataset pré-processado.
- [X] Gerar visualizações dos resultados do clustering.
- [X] Aplicar LIME para explicar as predições de cluster.
- [X] Aplicar SHAP para obter uma visão global da importância das features.
- [X] Salvar as visualizações e os resultados da interpretabilidade.

## 6. Consolidação e Documentação

- [ ] Criar a estrutura do notebook Jupyter (.ipynb) consolidado.
    - [ ] Seção de introdução e carregamento de dados.
    - [ ] Seção de pré-processamento.
    - [ ] Seção para K-Means Tradicional (K ótimo, resultados, visualizações, LIME, SHAP).
    - [ ] Seção para Robust and Sparse K-Means (parâmetros, resultados, visualizações, LIME, SHAP).
    - [ ] Seção para Adaptively Robust and Sparse K-Means (parâmetros, resultados, visualizações, LIME, SHAP).
    - [ ] Seção de conclusões e comparações.
- [ ] Integrar todos os códigos, resultados, visualizações e explicações no notebook Jupyter.
- [ ] Gerar todos os arquivos de imagem (.png/.jpg) das visualizações (gráficos de cotovelo, silhouette, Davies-Bouldin, scatter plots dos clusters, plots LIME, plots SHAP) e garantir que sejam referenciados corretamente no notebook e no relatório.
- [X] Escrever o relatório detalhado em formato Markdown, que será posteriormente convertido para PDF.
    - [X] Introdução ao problema e aos objetivos.
    - [X] Descrição do dataset e do pré-processamento realizado.
    - [X] Explicação detalhada das métricas de avaliação de clustering: Método do Cotovelo, Silhouette Score e Davies-Bouldin Index.
    - [X] Para cada algoritmo de clustering:
        - [X] Breve descrição do algoritmo.
        - [X] Processo de determinação do K ótimo e outros hiperparâmetros (com justificativas e explicações de como os valores foram obtidos).
        - [X] Apresentação dos resultados do clustering (com visualizações).
        - [X] Análise e interpretação dos agrupamentos utilizando LIME e SHAP (com visualizações e explicação do peso e influência das features).
    - [X] Comparação dos resultados dos três algoritmos.
    - [X] Conclusões finais.
- [ ] Converter o relatório Markdown para PDF.

## 7. Entrega dos Resultados

- [ ] Verificar se todos os artefatos estão completos: notebook Jupyter (.ipynb), todos os arquivos de imagem (.png/.jpg) e o relatório final em PDF.
- [ ] Enviar todos os arquivos para o usuário através da ferramenta de mensagem, com uma breve descrição do conteúdo de cada um.
