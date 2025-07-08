import numpy as np

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

print(f"Filtro projetado com Janela de Hanning e M={M}")

print("\nCoeficientes Normalizados do Filtro h[n]:")
print(np.round(h_norm, 8).tolist())

print("\nEquação de Diferenças:")
equation = "y[n] = "
for i, coeff in enumerate(h_norm):
    if abs(coeff) > 1e-9:
        if i > 0:
            equation += f"{coeff:+.4f}x[n-{i}] "
        else:
            equation += f"{coeff:.4f}x[n-{i}] "

print(equation.replace("+ -", "- ").strip())