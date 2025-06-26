import numpy as np

# --- 1. Dados do Problema ---
# [cite_start]Sinal de fala s[n] e taxa de amostragem Fs [cite: 154]
s = np.array([1, 2, -3, 3, -2, 1, -1, -1, 4, 5, -5, 4])
Fs = 8000  # amostras por segundo

# [cite_start]Parâmetros da janela [cite: 155]
window_duration_s = 0.00025  # 0.25 ms
overlap = 0.50  # 50%

# --- 2. Cálculos da Janela ---
# Tamanho da janela em amostras
L = int(window_duration_s * Fs)

# Deslocamento (passo ou hop size) em amostras
hop_size = int(L * (1 - overlap))

# --- 3. Extração de Características ---
f_energy = []
f_zcr = []

# Determina o número de janelas que podem ser extraídas
num_windows = int(np.floor((len(s) - L) / hop_size)) + 1

# Itera sobre o sinal para extrair cada janela e calcular as características
for i in range(num_windows):
    # Define o início e o fim da janela atual
    start_index = i * hop_size
    end_index = start_index + L
    window = s[start_index:end_index]

    # [cite_start]a) Cálculo da Energia [cite: 135]
    # Fórmula: E = sum(s_i^2)
    energy = np.sum(window**2)
    f_energy.append(energy)

    # [cite_start]b) Cálculo do Zero-Crossing Rate (ZCR) [cite: 142]
    # Fórmula: ZCR = 0.5 * sum(|sign(s_j) - sign(s_j+1)|)
    # A função np.sign(0) é 0, mas a definição da aula é sign(x>=0) = 1.
    # Como não há zeros no sinal, np.sign funciona perfeitamente.
    # Para a janela de 2 amostras, a fórmula simplifica para:
    zcr = 0.5 * np.abs(np.sign(window[0]) - np.sign(window[1]))
    f_zcr.append(int(zcr)) # Converte para inteiro (0 ou 1)

# --- 4. Exibição dos Resultados ---
print("--- Resolução do Exercício ST10 ---")
print(f"\n1. Comprimento do vetor de características f[n]: {len(f_energy)}")
print(f"\n2. Valores de f[n] para Energia:")
print(f"   f_energia[n] = {f_energy}")
print(f"\n3. Valores de f[n] para ZCR:")
print(f"   f_ZCR[n] = {f_zcr}")