import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

import scipy.cluster.hierarchy as sch

# --- 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
print("--- Iniciando Pré-processamento ---")

try:
    df_full = pd.read_csv('datatran2023.csv', sep=';', encoding='latin-1', low_memory=False)
except FileNotFoundError:
    print("ERRO: Arquivo 'datatran2023.csv' não encontrado.")
    exit()

df_sp = df_full[df_full['uf'] == 'SP'].copy()

# Selecionando features relevantes para agrupamento + a coluna para avaliação externa
features_para_cluster = [
    'dia_semana', 'uso_solo', 'horario',
    'condicao_metereologica', 'tipo_pista'
]
coluna_avaliacao_externa = 'tipo_acidente'

df = df_sp[features_para_cluster + [coluna_avaliacao_externa]].copy()

# Tratando o horário
df['hora_do_dia'] = pd.to_datetime(df['horario'], format='%H:%M:%S').dt.hour
df = df.drop(columns=['horario'])

# Removendo linhas com dados ausentes para simplificar
df.dropna(inplace=True)

# Separando a coluna de avaliação externa antes do pré-processamento
y_true_labels = df[coluna_avaliacao_externa]
X = df.drop(columns=[coluna_avaliacao_externa])

# Identificando colunas numéricas e categóricas
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Criando um pipeline de pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Aplicando o pré-processamento
X_processed = preprocessor.fit_transform(X)
print("--- Pré-processamento Concluído ---")
print(f"Dimensões dos dados processados: {X_processed.shape}\n")


# --- 2. APLICAÇÃO DO K-MEANS ---
print("--- Iniciando Análise com K-Means ---")

# a) Método do Cotovelo (Elbow Method)
wcss = []
k_range = range(1, 11)
for i in k_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_processed)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('Método do Cotovelo (Elbow Method)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.savefig('elbow_plot.png')
plt.show()

# b) Coeficiente de Silhueta
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_processed)
    score = silhouette_score(X_processed, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"Para k={i}, Coeficiente de Silhueta é {score:.3f}")

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
plt.title('Coeficiente de Silhueta para Diferentes k')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.savefig('silhouette_scores.png')
plt.show()

# c) Treinando o modelo final com k=3 (baseado na análise anterior)
optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
kmeans_labels = kmeans_final.fit_predict(X_processed)
print(f"\nK-Means treinado com k={optimal_k}.")


# --- 3. APLICAÇÃO DO AGRUPAMENTO HIERÁRQUICO ---
print("\n--- Iniciando Análise com Agrupamento Hierárquico ---")

# a) Gerando o Dendrograma (usando uma amostra para não sobrecarregar a memória)
plt.figure(figsize=(15, 8))
sample_for_dendrogram = X_processed if X_processed.shape[0] < 5000 else X_processed[:5000]
dendrogram = sch.dendrogram(sch.linkage(sample_for_dendrogram.toarray(), method='ward'))
plt.title('Dendrograma (com amostra de dados)')
plt.xlabel('Acidentes')
plt.ylabel('Distância Euclidiana')
plt.axhline(y=100, color='r', linestyle='--') # Linha de corte exemplo
plt.savefig('dendrogram.png')
plt.show()

# b) Treinando o modelo com k=3
hierarchical_final = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward')
hierarchical_labels = hierarchical_final.fit_predict(X_processed.toarray())
print(f"Agrupamento Hierárquico treinado com k={optimal_k}.")


# --- 4. AVALIAÇÃO E DISCUSSÃO FINAL ---
print("\n--- Comparação Final dos Modelos ---")

# a) Validação Interna (Silhueta)
sil_kmeans = silhouette_score(X_processed, kmeans_labels)
sil_hierarchical = silhouette_score(X_processed, hierarchical_labels)
print(f"Coeficiente de Silhueta (K-Means): {sil_kmeans:.3f}")
print(f"Coeficiente de Silhueta (Hierárquico): {sil_hierarchical:.3f}")

# b) Validação Externa (Índice Rand Ajustado)
ari_kmeans = adjusted_rand_score(y_true_labels, kmeans_labels)
ari_hierarchical = adjusted_rand_score(y_true_labels, hierarchical_labels)
print(f"Índice Rand Ajustado (K-Means): {ari_kmeans:.3f}")
print(f"Índice Rand Ajustado (Hierárquico): {ari_hierarchical:.3f}")

# c) Visualização dos Clusters com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed.toarray())

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
# K-Means
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=10)
axes[0].set_title('Clusters K-Means (Visualização 2D com PCA)')
axes[0].set_xlabel('Componente Principal 1')
axes[0].set_ylabel('Componente Principal 2')

# Hierárquico
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='viridis', s=10)
axes[1].set_title('Clusters Hierárquico (Visualização 2D com PCA)')
axes[1].set_xlabel('Componente Principal 1')
axes[1].set_ylabel('Componente Principal 2')

plt.savefig('cluster_comparison.png')
plt.show()