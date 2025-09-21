# -*- coding: utf-8 -*-

# # Notebook: Análise de Acidentes em Rodovias Federais de São Paulo

# ## 1. Carregamento e Preparação Inicial dos Dados

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

# --- Carregamento dos Dados (LENDO O ARQUIVO LOCAL) ---
# O código agora lê o arquivo que você baixou e renomeou.
try:
    # O arquivo original da PRF usa codificação 'latin-1' e separador ';'.
    df_full = pd.read_csv('datatran2023.csv', sep=';', encoding='latin-1', low_memory=False)
except FileNotFoundError:
    print("ERRO: Arquivo 'datatran2023.csv' não encontrado.")
    print("Verifique se o nome do arquivo está correto e se ele está na mesma pasta do script.")
    exit() # Encerra o script se o arquivo não for encontrado

# --- Filtragem e Seleção de Colunas ---
# Filtrando os dados apenas para o estado de São Paulo
df_sp = df_full[df_full['uf'] == 'SP'].copy()

# Selecionando as colunas relevantes para a análise (nomes corretos para o arquivo de 2023)
relevant_cols = [
    'data_inversa', 'dia_semana', 'horario', 'classificacao_acidente',
    'condicao_metereologica', 'tipo_pista', 'tracado_via',
    'uso_solo', 'pessoas', 'mortos', 'feridos_leves', 'feridos_graves',
    'causa_acidente'
]
df = df_sp[relevant_cols].copy()

# --- Criação da Variável Alvo ---
# 1 se houve pelo menos 1 morto, 0 caso contrário.
df['houve_morte'] = (df['mortos'] > 0).astype(int)

# Removendo colunas que não serão usadas como features (preditores)
df = df.drop(columns=['mortos', 'classificacao_acidente'])

print("### Amostra do Dataset de Acidentes em SP (Dados Carregados com Sucesso):")
print(df.head())
print("\n" + "="*50 + "\n")

print("### Informações Gerais:")
df.info()
print("\n" + "="*50 + "\n")

print("### Estatísticas Descritivas:")
print(df.describe())
print("\n" + "="*50 + "\n")

print("### Distribuição da Variável Alvo ('houve_morte'):")
print(df['houve_morte'].value_counts(normalize=True))
print("\n" + "="*50 + "\n")


# ## 2. Análise Exploratória e Visualização de Dados

# Configurando o estilo dos gráficos
sns.set(style="whitegrid")

# Relação entre Condição Meteorológica e Acidentes com Morte
plt.figure(figsize=(12, 7))
sns.countplot(y='condicao_metereologica', hue='houve_morte', data=df, order=df['condicao_metereologica'].value_counts().index)
plt.title('Condição Meteorológica vs. Ocorrência de Morte')
plt.xlabel('Contagem de Acidentes')
plt.ylabel('Condição Meteorológica')
plt.show()

# Relação entre Dia da Semana e Acidentes com Morte
plt.figure(figsize=(12, 7))
dias_ordem = ['domingo', 'segunda-feira', 'terça-feira', 'quarta-feira', 'quinta-feira', 'sexta-feira', 'sábado']
sns.countplot(x='dia_semana', hue='houve_morte', data=df, order=dias_ordem)
plt.title('Dia da Semana vs. Ocorrência de Morte')
plt.xlabel('Dia da Semana')
plt.ylabel('Contagem de Acidentes')
plt.xticks(rotation=45)
plt.show()

# Correlação entre as variáveis numéricas
plt.figure(figsize=(10, 8))
corr_matrix = df[['pessoas', 'feridos_leves', 'feridos_graves', 'houve_morte']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação entre Variáveis Numéricas')
plt.show()


# ## 3. Pré-processamento dos Dados

# ### a. Discretização de Variável Contínua (Horário)

# Convertendo 'horario' para um formato de hora
df['hora_acidente'] = pd.to_datetime(df['horario'], format='%H:%M:%S').dt.hour

# Função para discretizar a hora em períodos do dia
def discretizar_periodo(hora):
    if 5 <= hora < 12:
        return 'Manhã'
    elif 12 <= hora < 18:
        return 'Tarde'
    elif 18 <= hora < 24:
        return 'Noite'
    else:
        return 'Madrugada'

df['periodo_dia'] = df['hora_acidente'].apply(discretizar_periodo)

print("### Discretização do Horário em Períodos do Dia:")
print(df[['horario', 'hora_acidente', 'periodo_dia']].head())
print("\n" + "="*50 + "\n")


# ### b. Encoding de Variáveis Categóricas

y_target = df['houve_morte']
df_to_process = df.drop(columns=['data_inversa', 'horario', 'hora_acidente', 'houve_morte'])

categorical_cols = df_to_process.select_dtypes(include=['object']).columns
numerical_cols = df_to_process.select_dtypes(include=np.number).columns

df_encoded = pd.get_dummies(df_to_process, columns=categorical_cols, drop_first=True)
X = df_encoded

print("### Dimensões do dataset de features após One-Hot Encoding:")
print(X.shape)
print("\n### Amostra dos dados com encoding:")
print(X.head())
print("\n" + "="*50 + "\n")


# ### c. Escalonamento de Atributos (Feature Scaling)

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print("### Amostra dos Dados Após Escalonamento (Padronização Z-score):")
print(X_scaled.head())
print("\n" + "="*50 + "\n")


# ## 4. Seleção de Atributos (Feature Selection)

# Para a seleção de atributos, usamos o dataframe antes do escalonamento (X)
mutual_info = mutual_info_classif(X, y_target, discrete_features=False)
mi_series = pd.Series(mutual_info, index=X.columns)
mi_series = mi_series.sort_values(ascending=False)

plt.figure(figsize=(12, 8))
mi_series.head(15).sort_values(ascending=True).plot(kind='barh')
plt.title('As 15 Features Mais Importantes (usando Informação Mútua)')
plt.xlabel('Informação Mútua')
plt.ylabel('Features')
plt.show()

print("### Ranking das 15 Features Mais Importantes:")
print(mi_series.head(15))
print("\n" + "="*50 + "\n")

print("--- FIM DA ANÁLISE ---")