# ST4 - Projeto de Filtro FIR Passa-Altas usando o método do professor (espelhamento e modulação)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from sympy import symbols, simplify

# Parâmetros do filtro
fs = 24000            # Frequência de amostragem
fc = 1500             # Frequência de corte
M = 5                 # Ordem do filtro
n = np.arange(0, M + 1)
center = M / 2

# Frequência angular complementada: Wc = pi - pi * (fc / (fs / 2)) = 7pi/8
wc = np.pi * (1 - (fc / (fs / 2)))

# Passo 1: gerar h[n] com sinc deslocada
h = np.where(
    n == center,
    wc / np.pi,
    np.sin(wc * (n - center)) / (np.pi * (n - center))
)

# Passo 2: espelhar o vetor h[n] (h[-n])
h_reversed = h[::-1]

# Passo 3: aplicar modulação alternada (+, -, +, -, ...)
modulated = np.array([val * (-1) ** i for i, val in enumerate(h_reversed)])

# Resultado final
print("Coeficientes do filtro passa-altas g[n]:")
print(np.round(modulated, 6))

# Transformada Z simbólica
g = modulated
z = symbols('z')
Gz = sum([g[i] * z**(-i) for i in range(len(g))])
Gz_simplified = simplify(Gz)
print("\nTransformada Z G(z):")
print(Gz_simplified)

# Resposta em frequência
ticks = [0, 1500, 6000, 12000]
labels = ["0", "1500 Hz", "6000 Hz (Nyquist)", "12000 Hz"]
w, h_response = freqz(g, worN=8000)
frequencies = w * fs / (2 * np.pi)

plt.figure(figsize=(10, 4))
plt.plot(frequencies, 20 * np.log10(np.abs(h_response)), label="|G(f)| em dB", color='orange')
plt.axvline(x=1500, color='red', linestyle='--', label='Frequência de corte (1500 Hz)')
plt.title("Resposta em Frequência do Filtro FIR Passa-Altas")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
