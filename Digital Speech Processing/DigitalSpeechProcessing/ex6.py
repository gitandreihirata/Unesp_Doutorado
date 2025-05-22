# ST6 - Cálculo Manual do Filtro FIR Band-Stop (Subtipo I)
# apostila (página 46)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import freqz


fs = 10000
fc1 = 2500
fc2 = 3500
M = 4
n = np.arange(M + 1)
center = M / 2


wc1 = np.pi * fc1 / (fs / 2)  # π/2
wc2 = np.pi * fc2 / (fs / 2)  # 0.7π


def sinc(w, n, center):
    result = np.zeros_like(n, dtype=float)
    for i in range(len(n)):
        x = n[i] - center
        if x == 0:
            result[i] = w / np.pi
        else:
            result[i] = np.sin(w * x) / (np.pi * x)
    return result


term_allpass = sinc(np.pi, n, center)
term_wc2 = sinc(wc2, n, center)
term_wc1 = sinc(wc1, n, center)


q = term_allpass - term_wc2 + term_wc1


q_normalizado = q / np.sum(q)


for i in range(len(n)):
    print(f"n = {n[i]} -> All-Pass = {term_allpass[i]:.6f}, Sinc(wc2) = {term_wc2[i]:.6f}, Sinc(wc1) = {term_wc1[i]:.6f}, q[n] = {q[i]:.6f}, q[n] norm = {q_normalizado[i]:.6f}")


title = "Resposta em Frequência - Filtro ST6"
w, h_response = freqz(q_normalizado, worN=8000)
frequencies = w * fs / (2 * np.pi)
plt.figure(figsize=(8, 4))
plt.plot(frequencies, 20 * np.log10(np.abs(h_response)))
plt.axvline(x=2500, color='red', linestyle='--', label='2500 Hz')
plt.axvline(x=3500, color='red', linestyle='--', label='3500 Hz')
plt.title(title)
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
