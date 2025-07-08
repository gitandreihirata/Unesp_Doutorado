import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

M = 22
omega_c = 0.25 * np.pi

alpha = omega_c / np.pi
center = M / 2.0

n = np.arange(M + 1)
h_ideal = np.zeros(M + 1)

for i in n:
    if i == center:
        h_ideal[i] = alpha
    else:
        h_ideal[i] = np.sin(np.pi * alpha * (i - center)) / (np.pi * (i - center))

w_hanning = np.hanning(M + 1)

h = h_ideal * w_hanning

h_norm = h / np.sum(h)

print("--- Exercício (2) com M=22 e Janela de Hanning ---")

print("\nCoeficientes Normalizados do Filtro h[n]:")
for i, coeff in enumerate(h_norm):
    print(f"h[{i:02d}] = {coeff:+.8f}")

print("\nEquação de Diferenças:")
print("y[n] = ")
is_first_term = True
for i, coeff in enumerate(h_norm):
    if abs(coeff) > 1e-9:
        if is_first_term:
            print(f"  {coeff:.8f} * x[n-{i}]")
            is_first_term = False
        else:
            print(f"  {coeff:+.8f} * x[n-{i}]")

w, H = freqz(h_norm, 1, worN=2048)

plt.figure(figsize=(12, 8))
plt.plot(w / np.pi, 20 * np.log10(np.abs(H)))
plt.title('Resposta de Frequência do Filtro Projetado (Hanning, M=22)')
plt.xlabel('Frequência Normalizada (x π rad/amostra)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.ylim(-80, 5)
plt.show()