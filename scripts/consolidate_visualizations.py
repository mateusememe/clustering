import os
import pandas as pd

# Diretórios onde os resultados dos algoritmos foram salvos
base_dir = "/home/ubuntu"
kmeans_dir = os.path.join(base_dir, "kmeans_results")
rskc_dir = os.path.join(base_dir, "rskc_results")
arskc_dir = os.path.join(base_dir, "arskc_results")

output_summary_file = os.path.join(base_dir, "visualizations_summary.md")

all_image_files = []

def list_files(directory, algorithm_name):
    files_info = []
    if not os.path.exists(directory):
        print(f"Diretório {directory} não encontrado.")
        return files_info
    for f in os.listdir(directory):
        if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".webp") or f.endswith(".svg"):
            files_info.append({"algorithm": algorithm_name, "filename": f, "path": os.path.join(directory, f)})
            all_image_files.append(os.path.join(directory, f))
    return files_info

kmeans_plots = list_files(kmeans_dir, "K-Means")
rskc_plots = list_files(rskc_dir, "RSKC")
arskc_plots = list_files(arskc_dir, "ARSKC")

summary_content = "# Resumo das Visualizações Geradas\n\n"

summary_content += "## K-Means Tradicional\n"
if kmeans_plots:
    for plot in kmeans_plots:
        summary_content += f"- Arquivo: `{plot['filename']}`\n"
else:
    summary_content += "- Nenhuma visualização encontrada para K-Means.\n"
summary_content += "\n"

summary_content += "## Robust and Sparse K-Means (RSKC)\n"
if rskc_plots:
    for plot in rskc_plots:
        summary_content += f"- Arquivo: `{plot['filename']}`\n"
else:
    summary_content += "- Nenhuma visualização encontrada para RSKC.\n"
summary_content += "\n"

summary_content += "## Adaptively Robust and Sparse K-Means (ARSKC)\n"
if arskc_plots:
    for plot in arskc_plots:
        summary_content += f"- Arquivo: `{plot['filename']}`\n"
else:
    summary_content += "- Nenhuma visualização encontrada para ARSKC.\n"
summary_content += "\n"

summary_content += "## Lista Consolidada de Arquivos de Imagem\n"
for img_path in all_image_files:
    summary_content += f"- `{img_path}`\n"

with open(output_summary_file, "w") as f:
    f.write(summary_content)

print(f"Resumo das visualizações salvo em: {output_summary_file}")
print("\nVisualizações K-Means:")
for plot in kmeans_plots:
    print(f"  - {plot['path']}")

print("\nVisualizações RSKC:")
for plot in rskc_plots:
    print(f"  - {plot['path']}")

print("\nVisualizações ARSKC:")
for plot in arskc_plots:
    print(f"  - {plot['path']}")

print("\nConsolidação e organização das visualizações concluída. Os arquivos estão prontos para serem referenciados no notebook e relatório.")

