import numpy as np
import math
from collections import Counter

s = np.array([-1, 2, -3, 3, 2, 1, -1, -1, -4, 5, 5, 4])
Fs = 16000
window_duration_s = 0.000125
overlap = 0.50
log_base = 10

L = int(window_duration_s * Fs)
hop_size = int(L * (1 - overlap))

f_entropy = []

num_windows = int(np.floor((len(s) - L) / hop_size)) + 1

for i in range(num_windows):
    start_index = i * hop_size
    end_index = start_index + L
    window = s[start_index:end_index]

    counts = Counter(window)
    entropy = 0.0

    for count in counts.values():
        p_i = count / L
        entropy -= p_i * math.log(p_i, log_base)

    f_entropy.append(entropy)

print("--- Prova Final: Exercício (5) ---")
print(f"\n1. Comprimento do vetor de características f[n]: {len(f_entropy)}")
print(f"\n2. Valores de f[n] para Entropia (base 10):")
print(np.round(f_entropy, 3).tolist())