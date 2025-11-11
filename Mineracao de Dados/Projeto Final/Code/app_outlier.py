from flask import Flask, request, jsonify
import joblib
import numpy as np
import warnings


warnings.filterwarnings('ignore', category=UserWarning)


app = Flask(__name__)

print("Carregando modelos e estatísticas...")
try:
    scaler = joblib.load('traffic_scaler.joblib')
    lof_model = joblib.load('lof_model.joblib')


    train_mean = scaler.mean_
    train_std = scaler.scale_

    print("Modelos e estatísticas carregados com sucesso.")
    print(f"-> Média (para Z-Score): {train_mean}")
    print(f"-> Desvio Padrão (para Z-Score): {train_std}")

except FileNotFoundError:
    print("ERRO CRÍTICO: Arquivos 'traffic_scaler.joblib' ou 'lof_model.joblib' não encontrados.")
    print("Por favor, execute 'python train_outlier_model.py' primeiro.")
    exit()
except Exception as e:
    print(f"Erro inesperado ao carregar modelos: {e}")
    exit()



Z_SCORE_THRESHOLD = 3.0 


def check_z_score_outlier(data_point):
    """
    Verifica se um ponto de dado é um outlier usando Z-Score.
    data_point: np.array de 1D, ex: [vehicleCount, avgTime, hora]
    """
    # Z-Score = (Valor - Média) / Desvio Padrão
    z_scores = (data_point - train_mean) / train_std


    is_outlier = np.any(np.abs(z_scores) > Z_SCORE_THRESHOLD)

    return bool(is_outlier), z_scores

def check_lof_outlier(data_point_scaled):
    """
    Verifica se um ponto de dado é um outlier usando o modelo LOF treinado.
    data_point_scaled: np.array de 2D já escalonado, ex: [[...]]
    """

    prediction = lof_model.predict(data_point_scaled)
    is_outlier = prediction[0] == -1


    score = lof_model.score_samples(data_point_scaled)[0]

    return bool(is_outlier), score


@app.route('/api/check_traffic_outlier', methods=['POST'])
def check_traffic_outlier():

    if not request.json:
        return jsonify({"error": "Nenhum dado JSON recebido."}), 400

    data = request.json

    try:

        vehicle_count = float(data['vehicleCount'])
        avg_time = float(data['avgTimeOnSensor'])
        hora = int(data['hora_do_dia'])


        data_point = np.array([vehicle_count, avg_time, hora])


        is_outlier_zscore, z_scores = check_z_score_outlier(data_point)

        data_point_scaled = scaler.transform(data_point.reshape(1, -1))
        is_outlier_lof, lof_score = check_lof_outlier(data_point_scaled)


        response = {
            "input_data": data, 
            "analysis": {
                "is_outlier_zscore": is_outlier_zscore,
                "is_outlier_lof": is_outlier_lof,      
                "z_scores": z_scores.tolist(),         
                "lof_score": lof_score,                
                "z_score_threshold": Z_SCORE_THRESHOLD
            }
        }


        print(f"Análise: Z-Score={is_outlier_zscore}, LOF={is_outlier_lof} (Score: {lof_score:.2f}) <- Dados: {data_point}")

        return jsonify(response), 200

    except KeyError as e:
        return jsonify({"error": f"JSON incompleto. Campo faltando: {e}"}), 400
    except Exception as e:
        print(f"ERRO durante a predição: {e}")
        return jsonify({"error": f"Erro interno no servidor: {e}"}), 500


if __name__ == '__main__':

    app.run(port=5003, host='0.0.0.0')