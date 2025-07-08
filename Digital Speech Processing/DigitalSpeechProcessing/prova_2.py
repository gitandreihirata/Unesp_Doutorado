import numpy as np

omega_p = 0.1 * np.pi
omega_s = 0.4 * np.pi
delta_omega = omega_s - omega_p

delta_t_norm = delta_omega / np.pi
M = int(np.ceil(3.0 / delta_t_norm))

omega_c = (omega_p + omega_s) / 2
alpha = omega_c / np.pi
center = M / 2

n = np.arange(M + 1)
h_ideal = np.zeros(M + 1)

for i in n:
    if i == center:
        h_ideal[i] = alpha
    else:
        h_ideal[i] = np.sin(np.pi * alpha * (i - center)) / (np.pi * (i - center))

w_barlett = np.bartlett(M + 1)

h = h_ideal * w_barlett

h_norm = h / np.sum(h)

print("--- Prova Final: Exercício (2) ---")
print(f"\nOrdem do Filtro (M) calculada: {M}")
print("\nCoeficientes Normalizados do Filtro h[n]:")
print(np.round(h_norm, 4))
print("\nEquação de Diferenças:")
equation = "y[n] = "
for i, coeff in enumerate(h_norm):
    if np.abs(coeff) > 1e-6:
        term = f"{coeff:+.4f}x[n-{i}] "
        equation += term
print(equation.replace("+ -", "- ").strip())
