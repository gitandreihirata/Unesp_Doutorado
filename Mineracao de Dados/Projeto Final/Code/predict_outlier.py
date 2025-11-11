# predict_outlier.py
import joblib
import numpy as np
import warnings
import sys  
import json  


warnings.filterwarnings('ignore', category=UserWarning)



Z_SCORE_THRESHOLD = 3.0  # Limite Z-Score



def check_z_score_outlier(data_point, train_mean, train_std):
    z_scores = (data_point - train_mean) / train_std
    is_outlier = np.any(np.abs(z_scores) > Z_SCORE_THRESHOLD)
    return bool(is_outlier), z_scores.tolist()


def check_lof_outlier(data_point_scaled, lof_model):
    prediction = lof_model.predict(data_point_scaled)
    is_outlier = prediction[0] == -1
    score = lof_model.score_samples(data_point_scaled)[0]
    return bool(is_outlier), score


def main():
    try:
 
        scaler = joblib.load('traffic_scaler.joblib')
        lof_model = joblib.load('lof_model.joblib')
        train_mean = scaler.mean_
        train_std = scaler.scale_

        # 2. Ler Argumentos da Linha de Comando
        # sys.argv[0] é o nome do script (predict_outlier.py)
        # sys.argv[1] será o vehicleCount
        # sys.argv[2] será o avgTimeOnSensor
        # sys.argv[3] será a hora_do_dia
        if len(sys.argv) != 4:
            print(json.dumps({"error": f"Uso incorreto. Esperava 3 argumentos, recebeu {len(sys.argv) - 1}"}))
            sys.exit(1)

        vehicle_count = float(sys.argv[1])
        avg_time = float(sys.argv[2])
        hora = int(sys.argv[3])

        data_point = np.array([vehicle_count, avg_time, hora])


        is_outlier_zscore, z_scores = check_z_score_outlier(data_point, train_mean, train_std)

        data_point_scaled = scaler.transform(data_point.reshape(1, -1))
        is_outlier_lof, lof_score = check_lof_outlier(data_point_scaled, lof_model)


        response = {
            "is_outlier_zscore": is_outlier_zscore,
            "is_outlier_lof": is_outlier_lof,
            "z_scores": z_scores,
            "lof_score": lof_score,
            "z_score_threshold": Z_SCORE_THRESHOLD
        }


        print(json.dumps(response))
        sys.exit(0)  # Sair com sucesso

    except FileNotFoundError:
        print(json.dumps({"error": "Modelos 'traffic_scaler.joblib' ou 'lof_model.joblib' não encontrados."}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Erro interno no script Python: {e}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()