FROM jupyter/all-spark-notebook:hadoop-3

USER root

# Atualiza o pip e instala dependências adicionais
RUN pip install --upgrade pip \
    && pip install numpy pandas scikit-learn shap scipy kagglehub

# Concede permissões sudo ao usuário jovyan
ENV GRANT_SUDO=yes
ENV JUPYTER_ENABLE_LAB=yes

# Define o diretório padrão de trabalho
WORKDIR /home/jovyan/work

# Ajusta permissões dos arquivos para usuário jovyan
RUN chown -R jovyan:users /home/jovyan/work

# Usuário padrão para execução do notebook
USER jovyan
