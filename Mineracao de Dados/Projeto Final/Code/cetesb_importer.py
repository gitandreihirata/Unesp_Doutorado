# cetesb_importer.py
import sys
import pandas as pd
from pymongo import MongoClient
import json

# Argumentos: 1=Nome da Estação, 2=Ano
try:
    station_name = sys.argv[1]
    year = sys.argv[2]


    print(f"Iniciando importação (simulada) para Estação: {station_name}, Ano: {year}")
    print("Lendo arquivo CSV de exemplo: 'dados_cetesb_pinheiros_2024.csv'...")

    try:
        df_full = pd.read_csv('dados_cetesb_pinheiros_2024.csv', sep=';', encoding='latin-1', header=7)
    except FileNotFoundError:
        print(json.dumps(
            {"success": False, "error": "Arquivo CSV 'dados_cetesb_pinheiros_2024.csv' não encontrado no servidor."}))
        sys.exit(1)

    df_full.columns = ['data', 'hora', 'mp10']


    is_24h = df_full['hora'] == '24:00'
    df_full.loc[is_24h, 'hora'] = '00:00'
    df = df_full[df_full['data'].str.contains(year, na=False)].copy()  # Filtra pelo ano

    df['data_hora'] = pd.to_datetime(df['data'] + ' ' + df['hora'], format='%d/%m/%Y %H:%M', utc=True)
    df.loc[is_24h, 'data_hora'] = df.loc[is_24h, 'data_hora'] + pd.Timedelta(days=1)
    df['mp10'] = pd.to_numeric(df['mp10'], errors='coerce')


    MONGO_URL = 'mongodb://digitalt_admin:dbZzyqe28%24%25@localhost:27017/digitalt_smartcitydb?authSource=digitalt_smartcitydb'
    client = MongoClient(MONGO_URL)
    db = client['digitalt_smartcitydb']
    collection = db['air_quality_data']

    collection.delete_many({"station_name": station_name, "timestamp": {"$gte": pd.Timestamp(f"{year}-01-01"),
                                                                        "$lte": pd.Timestamp(
   
    records = []
    for _, row in df.iterrows():
        records.append({
            "timestamp": row['data_hora'],
            "station_name": station_name,
            "pollutant": "MP10", 
            "value": row['mp10']
        })

    if len(records) > 0:
        collection.insert_many(records)

    client.close()

    print(
        json.dumps({"success": True, "message": f"Importação (simulada) concluída. {len(records)} registros salvos."}))
    sys.exit(0)

except Exception as e:
    print(json.dumps({"success": False, "error": f"Erro interno no importador Python: {e}"}))
    sys.exit(1)