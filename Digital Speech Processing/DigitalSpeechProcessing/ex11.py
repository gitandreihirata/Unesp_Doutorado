import numpy as np

s = np.array([0, 1, 3, 0, 1, 3, 0, 1, 3, 0])

max_lag = len(s)
amdf_values = []

for tau in range(max_lag):
    if tau == 0:
        amdf_values.append(0)
        continue

    s1 = s[:-tau]
    s2 = s[tau:]
    amdf = np.sum(np.abs(s1 - s2))
    amdf_values.append(amdf)

amdf_values = np.array(amdf_values)

T_samples = np.argmin(amdf_values[1:]) + 1

print("--- Resolução do Exercício ST11 em Python ---")
print(f"Sinal de entrada s[n]: {s.tolist()}")
print(f"Valores AMDF calculados (para lags de 0 a {max_lag - 1}): {amdf_values.tolist()}")
print(f"\nPeríodo (T) encontrado: {T_samples} amostras")
print(f"Frequência Fundamental (F0): r / {T_samples} Hz")