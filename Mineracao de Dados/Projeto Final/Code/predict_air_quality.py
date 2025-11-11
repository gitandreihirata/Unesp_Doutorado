# predict_air_quality.py
import joblib
import numpy as np
import sys
import json
import warnings


print("--- [DEBUG] predict_air_quality.py: Script iniciado ---", file=sys.stderr)


warnings.filterwarnings('ignore', category=UserWarning)


def main():
    try:

        print("--- [DEBUG] Carregando modelos... ---", file=sys.stderr)
        scaler = joblib.load('air_quality_scaler.joblib')
        classifier = joblib.load('air_quality_classifier.joblib')
        print("--- [DEBUG] Modelos carregados. ---", file=sys.stderr)


        if len(sys.argv) != 4:
            print(json.dumps({"error": f"Uso incorreto. Esperava 3 argumentos, recebeu {len(sys.argv) - 1}"}))
            sys.exit(1)

        hora = int(sys.argv[1])
        dia_semana = int(sys.argv[2])
        mes = int(sys.argv[3])

        data_point = np.array([hora, dia_semana, mes])

        data_point_scaled = scaler.transform(data_point.reshape(1, -1))

        prediction = classifier.predict(data_point_scaled)
        probabilities = classifier.predict_proba(data_point_scaled)

        confidence = np.max(probabilities[0])

        response = {
            "predicted_class": prediction[0],
            "confidence_score": confidence
        }

        print(json.dumps(response))
        sys.exit(0)

    except FileNotFoundError as e:
        print("--- [DEBUG] ERRO: Arquivos .joblib não encontrados ---", file=sys.stderr)
        print(json.dumps(
            {"error": "Modelos 'air_quality_scaler.joblib' ou 'air_quality_classifier.joblib' não encontrados."}))
        sys.exit(1)
    except Exception as e:

        print(f"--- [DEBUG] ERRO INTERNO: {e} ---", file=sys.stderr)

        print(json.dumps({"error": f"Erro interno no script Python: {e}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()