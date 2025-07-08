import numpy as np

M = 8
Fs = 22050
f_c1 = 2000
f_c2 = 4000

alpha1 = (2 * f_c1) / Fs
alpha2 = (2 * f_c2) / Fs

n = np.arange(M + 1)
h = np.zeros(M + 1)
h_hp = np.zeros(M + 1)

center = M / 2
for i in n:
    if i == center:
        h[i] = alpha1
        h_hp[i] = alpha2
    else:
        h[i] = np.sin(np.pi * alpha1 * (i - center)) / (np.pi * (i - center))
        h_hp[i] = np.sin(np.pi * alpha2 * (i - center)) / (np.pi * (i - center))

d = np.zeros(M + 1)
d[int(center)] = 1

g = d - h_hp
q = h + g

normalization_factor = np.sum(q)
q_norm = q / normalization_factor

print("--- Prova Final: Exercício (1) ---")
print("\nCoeficientes Normalizados do Filtro q[n]:")
print(np.round(q_norm, 4))
print("\nEquação de Diferenças:")
equation = "y[n] = "
for i, coeff in enumerate(q_norm):
    term = f"{coeff:+.4f}x[n-{i}] "
    equation += term
print(equation.replace("+ -", "- "))