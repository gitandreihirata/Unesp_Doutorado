import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import f1_score, confusion_matrix

print("--- Iniciando Pré-processamento ---")

try:
    df_full = pd.read_csv('dados_cetesb_pinheiros_2024.csv', sep=';', encoding='latin-1', header=7)
except FileNotFoundError:
    print("ERRO: Arquivo 'dados_cetesb_pinheiros_2024.csv' não encontrado.")
    exit()

df_full.columns = ['data', 'hora', 'mp10']
df = df_full.copy()

is_24h = df['hora'] == '24:00'
df.loc[is_24h, 'hora'] = '00:00'
df['data_hora'] = pd.to_datetime(df['data'] + ' ' + df['hora'], format='%d/%m/%Y %H:%M')
df.loc[is_24h, 'data_hora'] = df.loc[is_24h, 'data_hora'] + pd.Timedelta(days=1)

df = df.set_index('data_hora')
df = df.drop(columns=['data', 'hora'])
df['mp10'] = pd.to_numeric(df['mp10'], errors='coerce')

df['mes'] = df.index.month
df['dia_semana'] = df.index.dayofweek
df['hora_do_dia'] = df.index.hour

def get_iqa_cat(mp10_val):
    if pd.isna(mp10_val): return None
    if mp10_val <= 50: return 'Boa'
    elif mp10_val <= 100: return 'Moderada'
    elif mp10_val <= 150: return 'Ruim'
    else: return 'Muito Ruim'

df['IQA_cat'] = df['mp10'].apply(get_iqa_cat)
df['mp10'] = df['mp10'].interpolate(method='linear')
df.dropna(subset=['IQA_cat'], inplace=True)

X = df.drop(columns=['IQA_cat', 'mp10'])
y = df['IQA_cat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
print(f"Tamanho do treino: {len(X_train)} | Tamanho do teste: {len(X_test)}")
print("--- Pré-processamento Concluído ---\n")

# --- 2. PARTE 1: COMPARAÇÃO DE ALGORITMOS ---
print("--- Iniciando Parte 1: Comparação de Modelos ---")

models = {
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "Bagging": BaggingClassifier(n_estimators=50, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average='weighted')
    results[name] = score
    print(f"Modelo: {name} | F1-Score (Ponderado): {score:.3f}")

best_model_name = max(results, key=results.get)
print(f"\nMelhor Modelo: {best_model_name}")

best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred_best, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f'Matriz de Confusão - {best_model_name}')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')
plt.savefig('matriz_confusao.png')
plt.show()

print("\n--- Otimizando Hiperparâmetros do Random Forest ---")
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
best_score = f1_score(y_test, y_pred_best_rf, average='weighted')
results['Random Forest Otimizado'] = best_score
print(f"Melhores Parâmetros: {grid_search.best_params_}")
print(f"F1-Score Otimizado: {best_score:.3f}")
print("--- Fim da Parte 1 ---\n")

# --- 3. PARTE 2: DECOMPOSIÇÃO OVR vs OVO ---
print("--- Iniciando Parte 2: Decomposição OVR vs OVO ---")

tree = DecisionTreeClassifier(random_state=42)
ovr_clf = OneVsRestClassifier(tree)
ovr_clf.fit(X_train, y_train)
y_pred_ovr = ovr_clf.predict(X_test)
score_ovr = f1_score(y_test, y_pred_ovr, average='weighted')
results['OVR (Árvore)'] = score_ovr
print(f"F1-Score com One-vs-Rest (OVR): {score_ovr:.3f}")

ovo_clf = OneVsOneClassifier(tree)
ovo_clf.fit(X_train, y_train)
y_pred_ovo = ovo_clf.predict(X_test)
score_ovo = f1_score(y_test, y_pred_ovo, average='weighted')
results['OVO (Árvore)'] = score_ovo
print(f"F1-Score com One-vs-One (OVO): {score_ovo:.3f}")
print("--- Fim da Parte 2 ---\n")

# --- 4. GERAÇÃO DO GRÁFICO COMPARATIVO FINAL ---
print("--- Gerando gráfico comparativo final ---")

results_df = pd.DataFrame(list(results.items()), columns=['Modelo', 'F1-Score']).sort_values('F1-Score', ascending=False)

plt.figure(figsize=(12, 8))
# #############################################################################
# CORREÇÃO APLICADA AQUI:
# 1. Adicionado hue='Modelo' e legend=False para corrigir o FutureWarning.
# 2. Corrigida a chamada da função ax.annotate.
# #############################################################################
ax = sns.barplot(x='F1-Score', y='Modelo', data=results_df, palette='viridis', hue='Modelo', legend=False)
plt.title('Comparação de Desempenho dos Modelos e Estratégias', fontsize=16)
plt.xlabel('F1-Score (Ponderado)', fontsize=12)
plt.ylabel('Modelo / Estratégia', fontsize=12)
plt.xlim(0, 1.0) # Ajuste do limite do eixo X

# Corrigindo o loop para adicionar os valores nas barras
for p in ax.patches:
    ax.annotate(f"{p.get_width():.3f}",
                xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                xytext=(15, 0),
                textcoords='offset points',
                ha='left',
                va='center')

plt.tight_layout()
plt.savefig('comparacao_desempenho.png')
plt.show()