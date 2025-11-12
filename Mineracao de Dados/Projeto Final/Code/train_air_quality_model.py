# train_air_quality_model.py
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


# Baseado nos padrões CETESB/CONAMA para MP10 (µg/m³)
# 0-50: Boa
# 51-100: Moderada
# > 100: Alerta (Ruim, Muito Ruim, Péssima)
def define_classe_qualidade(valor):
    if valor <= 50:
        return 'Boa'
    elif valor <= 100:
        return 'Moderada'
    else:
        return 'Alerta' # Agrupando Ruim, Muito Ruim, Péssima


print("Conectando ao MongoDB...")
MONGO_URL = 'mongodb://x'
DB_NAME = 'digitalt_smartcitydb'

COLLECTION_NAME = 'air_quality_data'

try:
    client = MongoClient(MONGO_URL)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Filtra apenas por 'mp10' (nosso primeiro modelo)
    dados_mongo = list(collection.find({"pollutant": "mp10"}))

    if len(dados_mongo) < 100: 
        print(f"ERRO: Encontrados apenas {len(dados_mongo)} registros de MP10. Importe mais dados.")
        exit()

    print(f"Dados carregados: {len(dados_mongo)} registros de MP10 encontrados.")
    df = pd.DataFrame(dados_mongo)
    client.close()

except Exception as e:
    print(f"Erro ao conectar ou buscar no MongoDB: {e}")
    exit()

# --- Features ---
print("Processando dados e criando features...")

df['timestamp'] = pd.to_datetime(df['timestamp'])

# Prever a qualidade baseado na hora e dia
df['hora_do_dia'] = df['timestamp'].dt.hour
df['dia_da_semana'] = df['timestamp'].dt.weekday # 0=Segunda, 6=Domingo
df['mes'] = df['timestamp'].dt.month


df['classe_qualidade'] = df['value'].apply(define_classe_qualidade)


# Features (X): Prever a qualidade com base no CONTEXTO (tempo)
X = df[['hora_do_dia', 'dia_da_semana', 'mes']]

y = df['classe_qualidade']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("Distribuição das classes ANTES do SMOTE:")
print(y_train.value_counts())


print("Treinando o StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


joblib.dump(scaler, 'air_quality_scaler.joblib')
print("Scaler salvo em 'air_quality_scaler.joblib'")


print("Aplicando SMOTE para balancear classes...")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

print("Distribuição das classes DEPOIS do SMOTE:")
print(pd.Series(y_resampled).value_counts())

print("Treinando o Classificador (RandomForest)...")

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_resampled, y_resampled)


joblib.dump(classifier, 'air_quality_classifier.joblib')
print("Classificador salvo em 'air_quality_classifier.joblib'")

print("\nAvaliando modelo nos dados de teste (não vistos)...")
y_pred = classifier.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

print("\n--- Treinamento Concluído! ---")
print("Modelos 'air_quality_scaler.joblib' e 'air_quality_classifier.joblib' estão prontos.")