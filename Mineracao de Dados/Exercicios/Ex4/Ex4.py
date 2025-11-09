import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
import sys  # Para garantir a codificação correta da saída

# Configura a saída padrão para UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Ignora avisos
warnings.filterwarnings('ignore')


# --- 1. FUNÇÕES DE SUPORTE (DEFINIÇÕES) ---

def define_classe_qualidade(valor):
    """ Define a classe de qualidade do ar (MP10) baseado nos padrões CETESB. """
    if valor <= 50:
        return 'Boa'
    elif valor <= 100:
        return 'Moderada'
    else:
        return 'Alerta'  # Agrupando Ruim, Muito Ruim, Péssima


def plotar_distribuicao_classes(df, filename):
    """ Salva um gráfico de barras da distribuição das classes. """
    plt.figure(figsize=(8, 5))
    sns.countplot(x='classe_qualidade', data=df, order=['Boa', 'Moderada', 'Alerta'])
    plt.title(f'Distribuição de Classes ({filename.split("_")[2]})')
    plt.xlabel('Classe de Qualidade do Ar')
    plt.ylabel('Contagem')
    plt.savefig(filename)
    print(f"Gráfico salvo: {filename}")
    plt.close()


def plotar_matriz_confusao(y_true, y_pred, classes, filename):
    """ Salva um gráfico de heatmap da matriz de confusão. """
    # Garante que as labels na matriz de confusão sigam a ordem das classes
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Matriz de Confusão ({filename.split("_")[2]})')
    plt.xlabel('Previsto')
    plt.ylabel('Verdadeiro')
    plt.savefig(filename)
    print(f"Gráfico salvo: {filename}")
    plt.close()


# --- 2. FUNÇÃO PRINCIPAL (EXECUÇÃO) ---

def main():
    print("--- Iniciando Análise de Dados Desbalanceados (Atividade 4) ---")

    # --- 2.1 CARREGAMENTO E LIMPEZA ---
    try:
        # Carrega os dados (ignora cabeçalho de 9 linhas, usa colunas 0,1,2)
        df = pd.read_csv(
            'qualar_2024_bauru_mp10.csv',
            sep=';',
            encoding='latin-1',
            skiprows=9,
            header=None,
            usecols=[0, 1, 2]
        )
        df.columns = ['Data', 'Hora', 'value']
    except FileNotFoundError:
        print("ERRO: Arquivo 'qualar_2024_bauru_mp10.csv' não encontrado.")
        print("Por favor, salve o arquivo CSV (que eu gerei) na mesma pasta deste script.")
        return
    except Exception as e:
        print(f"Erro ao ler o CSV: {e}")
        return

    # Limpeza
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df.dropna(subset=['Data', 'Hora', 'value'], inplace=True)

    is_24h = df['Hora'] == '24:00'
    df.loc[is_24h, 'Hora'] = '00:00'

    df['timestamp'] = pd.to_datetime(df['Data'] + ' ' + df['Hora'], format='%d/%m/%Y %H:%M', utc=True)
    df.loc[is_24h, 'timestamp'] = df.loc[is_24h, 'timestamp'] + pd.Timedelta(days=1)

    print(f"Total de {len(df)} registros válidos carregados.")

    # --- 2.2 ENGENHARIA DE FEATURES ---
    df['hora_do_dia'] = df['timestamp'].dt.hour
    df['dia_da_semana'] = df['timestamp'].dt.weekday
    df['mes'] = df['timestamp'].dt.month
    df['classe_qualidade'] = df['value'].apply(define_classe_qualidade)

    # Garante a ordem correta das classes para os gráficos
    classes_ordenadas = ['Boa', 'Moderada', 'Alerta']

    # Remove classes que podem não estar presentes no dataset de amostra (se for o caso)
    classes_presentes = [c for c in classes_ordenadas if c in df['classe_qualidade'].unique()]

    # Plota a distribuição ANTES
    plotar_distribuicao_classes(df, 'distribuicao_classes_antes.png')

    # --- 2.3 PREPARAÇÃO DOS DADOS ---
    X = df[['hora_do_dia', 'dia_da_semana', 'mes']]
    y = df['classe_qualidade']

    # Divisão Treino/Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Escalonamento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 2.4 TÉCNICA 1: BASELINE (SEM SMOTE) ---
    print("\n--- [TÉCNICA 1: Baseline (Sem SMOTE)] ---")
    rf_base = RandomForestClassifier(random_state=42)
    rf_base.fit(X_train_scaled, y_train)
    y_pred_base = rf_base.predict(X_test_scaled)

    print("Relatório de Classificação (Sem SMOTE):")
    print(classification_report(y_test, y_pred_base, labels=classes_presentes, zero_division=0))

    plotar_matriz_confusao(y_test, y_pred_base, classes_presentes, 'matriz_confusao_sem_smote.png')

    # --- 2.5 TÉCNICA 2: COM SMOTE ---
    print("\n--- [TÉCNICA 2: Com SMOTE] ---")

    # Aplica SMOTE
    print("Aplicando SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=min(len(y_train[y_train == 'Alerta']) - 1, 5))  # k_neighbors seguro
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
    print("SMOTE concluído.")

    # Plota a distribuição DEPOIS
    plotar_distribuicao_classes(pd.DataFrame(y_resampled, columns=['classe_qualidade']),
                                'distribuicao_classes_depois.png')

    # Treina o novo modelo
    rf_smote = RandomForestClassifier(random_state=42)
    rf_smote.fit(X_resampled, y_resampled)
    y_pred_smote = rf_smote.predict(X_test_scaled)

    print("Relatório de Classificação (Com SMOTE):")
    print(classification_report(y_test, y_pred_smote, labels=classes_presentes, zero_division=0))

    plotar_matriz_confusao(y_test, y_pred_smote, classes_presentes, 'matriz_confusao_com_smote.png')

    print("\n--- Análise Concluída ---")


if __name__ == "__main__":
    main()