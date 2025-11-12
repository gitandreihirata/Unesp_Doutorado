# qualar_importer.py
import sys
import pandas as pd
from pymongo import MongoClient
import json
import os  


def connect_to_db():
    MONGO_URL = 'mongodb://x'
    client = MongoClient(MONGO_URL)
    db = client['digitalt_smartcitydb']
    return db, client


def main():
    try:

        if len(sys.argv) != 4:
            print(json.dumps(
                {"success": False, "error": "Uso incorreto. Esperava 3 argumentos (ano, estacao, parametro)"}))
            sys.exit(1)

        year = int(sys.argv[1])
        station = sys.argv[2]
        parameter = sys.argv[3]  # ex: "mp10"


        db, client = connect_to_db()
        catalog_collection = db['qualar_file_catalog']

        file_info = catalog_collection.find_one({
            "year": year,
            "station": station,
            "parameter": parameter
        })

        if not file_info:
            client.close()
            print(json.dumps({"success": False, "error": "Arquivo não encontrado no catálogo."}))
            sys.exit(1)

        csv_path = file_info['full_path']
        csv_filename = file_info['filename']

        if not os.path.exists(csv_path):
            client.close()
            print(json.dumps(
                {"success": False, "error": f"Arquivo {csv_filename} não encontrado no caminho {csv_path}."}))
            sys.exit(1)

        try:
            df = pd.read_csv(
                csv_path,
                sep=';',
                encoding='latin-1',
                skiprows=9,  
                header=None,  
                usecols=[0, 1, 2]  
            )
        except pd.errors.EmptyDataError:
          
            client.close()
            print(json.dumps(
                {"success": True, "message": "Importação concluída. 0 registros de dados encontrados no arquivo.",
                 "file": csv_filename}))
            sys.exit(0)

  
        df.columns = ['Data', 'Hora', 'value']


        df['value'] = pd.to_numeric(df['value'], errors='coerce')  # Converte para número, - vira NaN

        df.dropna(subset=['Data', 'Hora', 'value'], inplace=True)  # Remove linhas com dados faltando

        if df.empty:
            client.close()
            print(json.dumps({"success": True, "message": "Importação concluída. 0 registros válidos após limpeza.",
                              "file": csv_filename}))
            sys.exit(0)



        is_24h = df['Hora'] == '24:00'
        df.loc[is_24h, 'Hora'] = '00:00'

  
        df['timestamp'] = pd.to_datetime(df['Data'] + ' ' + df['Hora'], format='%d/%m/%Y %H:%M', utc=True)

        df.loc[is_24h, 'timestamp'] = df.loc[is_24h, 'timestamp'] + pd.Timedelta(days=1)

        collection_name = 'air_quality_data'
        collection = db[collection_name]

        collection.delete_many({
            "station_name": station,
            "pollutant": parameter,
            "timestamp": {
                "$gte": pd.Timestamp(f"{year}-01-01T00:00:00Z"),
                "$lte": pd.Timestamp(f"{year}-12-31T23:59:59Z")
            }
        })

        records = []
        for _, row in df.iterrows():
            records.append({
                "timestamp": row['timestamp'],
                "station_name": station,
                "pollutant": parameter,
                "value": row['value']  
            })

        if len(records) > 0:
            collection.insert_many(records)

        client.close()
        print(json.dumps({"success": True, "message": f"Importação concluída. {len(records)} registros salvos.",
                          "file": csv_filename}))
        sys.exit(0)

    except Exception as e:
        if 'client' in locals():
            client.close()

        print(json.dumps({"success": False, "error": f"Erro interno no importador Python: {e}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()