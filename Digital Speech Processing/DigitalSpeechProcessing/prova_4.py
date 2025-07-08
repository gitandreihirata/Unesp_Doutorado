import numpy as np
from scipy.signal import zpk2tf

poles = np.array([-1/4, 1/3])
zeros = np.array([1/5])
gain = 1

num_poly_z, den_poly_z = zpk2tf(zeros, poles, gain)

b_coeffs = np.array([0, 1, -1/5])
a_coeffs = np.array([1, -1/12, -1/12])

term_y1 = -a_coeffs[1]/a_coeffs[0]
term_y2 = -a_coeffs[2]/a_coeffs[0]
term_x1 = b_coeffs[1]/a_coeffs[0]
term_x2 = b_coeffs[2]/a_coeffs[0]

is_stable_and_causal = np.all(np.abs(poles) < 1)

print("--- Prova Final: Exercício (4) ---")
print(f"\nPolos: {poles.tolist()}")
print(f"Zeros: {zeros.tolist()}")

print("\nFunção de Transferência H(z):")
print(f"Numerador (em z^-1): {b_coeffs.tolist()}")
print(f"Denominador (em z^-1): {np.round(a_coeffs, 4).tolist()}")

print("\nEquação de Diferenças:")
equation = f"y[n] = {term_y1:.4f}*y[n-1] + {term_y2:.4f}*y[n-2] + {term_x1:.4f}*x[n-1] {term_x2:+.4f}*x[n-2]"
print(equation.replace("+ -", "- "))


print("\nAnálise de Estabilidade e Causalidade:")
print(f"Magnitudes dos polos: {np.abs(poles).tolist()}")
print(f"Estável e Causal? {'Sim, todos os polos estão dentro do círculo unitário.' if is_stable_and_causal else 'Não'}")