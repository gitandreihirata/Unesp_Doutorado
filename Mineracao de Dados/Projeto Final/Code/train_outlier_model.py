# train_outlier_model.py
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


print("Conectando ao MongoDB...")
MONGO_URL = 'mongodb://x'
DB_NAME = 'digitalt_smartcitydb'
COLLECTION_NAME = 'sensor_vehicle_flow' 

try:
    client = MongoClient(MONGO_URL)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]


    dados_mongo = list(collection.find().limit(10000))

    if len(dados_mongo) < 50:
        print(f"ERRO: Encontrados apenas {len(dados_mongo)} registros. Rode a simulação por mais tempo.")
        exit()

    print(f"Dados carregados: {len(dados_mongo)} registros encontrados.")
    df = pd.DataFrame(dados_mongo)
    client.close()

except Exception as e:
    print(f"Erro ao conectar ou buscar no MongoDB: {e}")
    exit()


print("Processando dados...")


try:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
except Exception as e:
    print(f"Erro crítico ao tentar converter coluna 'timestamp': {e}.")
    exit()


df.dropna(subset=['timestamp'], inplace=True)

if df.empty:
    print("ERRO: Nenhum dado restou após limpar timestamps inválidos.")
    exit()


df['hora_do_dia'] = df['timestamp'].dt.hour


features_df = df[['vehicleCount', 'avgTimeOnSensor', 'hora_do_dia']].copy()


features_df.dropna(inplace=True)

if features_df.empty:
    print("ERRO: Nenhum dado válido após limpeza. Verifique os dados da simulação.")
    exit()

print(f"Dados prontos para treino: {len(features_df)} registros válidos.")
print("Amostra das features:")
print(features_df.head())

print("Treinando o StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_df)

joblib.dump(scaler, 'traffic_scaler.joblib')
print("Scaler salvo em 'traffic_scaler.joblib'")


print("Treinando o modelo Local Outlier Factor (LOF)...")

lof_model = LocalOutlierFactor(contamination=0.1, novelty=True)
lof_model.fit(X_scaled)


joblib.dump(lof_model, 'lof_model.joblib')
print("Modelo LOF salvo em 'lof_model.joblib'")

print("\n--- Treinamento Concluído! ---")
print("Arquivos 'traffic_scaler.joblib' e 'lof_model.joblib' foram criados.")
print("Estamos prontos para a Fase 2 (construir a API de detecção).")