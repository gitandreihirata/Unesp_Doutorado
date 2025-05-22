# ST5 - Filtro FIR Band-Stop seguindo o método do professor

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from sympy import symbols, simplify

# Parâmetros do filtro
fs = 10000  # Frequência de amostragem (Hz)
fc1 = 2500  # Frequência de corte inferior (Hz)
fc2 = 3500  # Frequência de corte superior (Hz)
M = 5       # Ordem do filtro
n = np.arange(M + 1)
center = M / 2

# Frequências angulares normalizadas
wc1 = 2 * np.pi * fc1 / fs
wc2 = 2 * np.pi * fc2 / fs

# Passo 1: filtro passa-baixa h[n]
h = np.where(
    n == center,
    wc1 / np.pi,
    np.sin(wc1 * (n - center)) / (np.pi * (n - center))
)

# Passo 2: filtro passa-alta g[n] por espelhamento e modulação
wc_mirror = np.pi - wc2
h_hp = np.where(
    n == center,
    wc_mirror / np.pi,
    np.sin(wc_mirror * (n - center)) / (np.pi * (n - center))
)
g = np.array([val * (-1)**i for i, val in enumerate(h_hp[::-1])])

# Passo 3: filtro Band-Stop final
q = h + g

# Exibir coeficientes
print("Coeficientes do filtro Passa-baixa (h):")
print(np.round(h, 6))

print("\nCoeficientes do filtro Passa-alta (g):")
print(np.round(g, 6))

print("\nCoeficientes do filtro Band-Stop q[n] (g + h):")
print(np.round(q, 6))

# Transformada Z simbólica
z = symbols('z')
Qz = sum([q[i] * z**(-i) for i in range(len(q))])
Qz_simplified = simplify(Qz)
print("\nTransformada Z Q(z):")
print(Qz_simplified)

# Resposta em frequência
w, h_response = freqz(q, worN=8000)
frequencies = w * fs / (2 * np.pi)

plt.figure(figsize=(10, 4))
plt.plot(frequencies, 20 * np.log10(np.abs(h_response)), label="Filtro Band-Stop", color='green')
plt.axvline(x=2500, color='red', linestyle='--', label='2500 Hz (limite inferior)')
plt.axvline(x=3500, color='red', linestyle='--', label='3500 Hz (limite superior)')
plt.title("Resposta em Frequência - Filtro FIR Band-Stop (Método do Professor)")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
