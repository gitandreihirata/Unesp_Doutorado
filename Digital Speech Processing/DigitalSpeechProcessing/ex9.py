import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, zpk2tf, tf2zpk

# --- 1. Definição de Polos e Zeros ---
poles = np.array([1/3, -1/6])
zeros = np.array([1/2])
gain = 1 # Assumindo ganho K=1

print(f"Polos: {poles}")
print(f"Zeros: {zeros}")
print(f"Ganho K: {gain}")

# --- 2. Função de Transferência H(z) ---
# H(z) = K * product(z - zi) / product(z - pi)
# Convertendo de formato zero-polo-ganho (zpk) para coeficientes de polinômios em z (tf)
# num_poly_z: coeficientes do numerador em potências decrescentes de z
# den_poly_z: coeficientes do denominador em potências decrescentes de z
num_poly_z, den_poly_z = zpk2tf(zeros, poles, gain)

print(f"\nCoeficientes do numerador (polinômio em z): {num_poly_z}")
print(f"Coeficientes do denominador (polinômio em z): {den_poly_z}")

# Para obter a forma para equação de diferenças (potências de z^-1)
# H(z) = (num_poly_z[0]*z + num_poly_z[1]) / (den_poly_z[0]*z^2 + den_poly_z[1]*z + den_poly_z[2])
# Dividimos por z^2 (maior potência no denominador)
# H(z) = (num_poly_z[0]*z^-1 + num_poly_z[1]*z^-2) / (den_poly_z[0] + den_poly_z[1]*z^-1 + den_poly_z[2]*z^-2)
# Coeficientes b (numerador para z^-1) e a (denominador para z^-1)
# Adicionando um zero à esquerda no numerador se necessário para igualar ordens para a conversão
if len(num_poly_z) < len(den_poly_z):
    num_poly_z_padded = np.concatenate(([0]*(len(den_poly_z) - len(num_poly_z) -1), num_poly_z)) # Ajuste para que b0 seja para z^0
else:
     num_poly_z_padded = num_poly_z

# Coeficientes para H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
# No nosso caso H(z) = (z^-1 - 0.5z^-2) / (1 - (1/6)z^-1 - (1/18)z^-2)
# Portanto, b = [0, 1, -0.5] e a = [1, -1/6, -1/18] se b0 é o termo de x[n]
# Ou H(z) = Y(z)/X(z)
# Y(z)(1 - (1/6)z^-1 - (1/18)z^-2) = X(z)(z^-1 - 0.5z^-2)
# y[n] - (1/6)y[n-1] - (1/18)y[n-2] = x[n-1] - 0.5x[n-2]
b_coeffs_z_inv = np.array([0, 1, -0.5]) # Coefs de x[n], x[n-1], x[n-2]
a_coeffs_z_inv = np.array([1, -1/6, -1/18])# Coefs de y[n], y[n-1], y[n-2]

print(f"\nCoeficientes b (numerador para z^-1): {b_coeffs_z_inv} (para x[n], x[n-1], x[n-2])")
print(f"Coeficientes a (denominador para z^-1): {a_coeffs_z_inv} (para y[n], y[n-1], y[n-2])")

# --- 3. Equação de Diferenças ---
# y[n] = (1/6)y[n-1] + (1/18)y[n-2] + x[n-1] - (1/2)x[n-2]
print("\n--- Equação de Diferenças ---")
eq_str = f"y[n] = ({a_coeffs_z_inv[1]:.4f})*y[n-1] + ({a_coeffs_z_inv[2]:.4f})*y[n-2] " \
         f"+ ({b_coeffs_z_inv[1]:.4f})*x[n-1] + ({b_coeffs_z_inv[2]:.4f})*x[n-2]"
# Corrigindo os sinais para a forma y[n] = ...
# a[0]y[n] + a[1]y[n-1] + a[2]y[n-2] = b[0]x[n] + b[1]x[n-1] + b[2]x[n-2]
# y[n] = (-a[1]/a[0])y[n-1] + (-a[2]/a[0])y[n-2] + (b[0]/a[0])x[n] + (b[1]/a[0])x[n-1] + (b[2]/a[0])x[n-2]
term_y1 = -a_coeffs_z_inv[1]/a_coeffs_z_inv[0]
term_y2 = -a_coeffs_z_inv[2]/a_coeffs_z_inv[0]
term_x0 = b_coeffs_z_inv[0]/a_coeffs_z_inv[0] # b0 é 0
term_x1 = b_coeffs_z_inv[1]/a_coeffs_z_inv[0]
term_x2 = b_coeffs_z_inv[2]/a_coeffs_z_inv[0]

final_eq_str = f"y[n] = {term_y1:.4f}*y[n-1] + {term_y2:.4f}*y[n-2] + {term_x1:.4f}*x[n-1] + {term_x2:.4f}*x[n-2]"
print(final_eq_str.replace("+ -", "- "))


# --- 4. Estabilidade e Causalidade ---
print("\n--- Estabilidade e Causalidade ---")
magnitudes_poles = np.abs(poles)
print(f"Magnitudes dos polos: {magnitudes_poles}")

is_stable = np.all(magnitudes_poles < 1)
is_causal_if_stable = is_stable # Para IIR, se os polos estão dentro do circulo unitario, um ROC causal garante estabilidade.

if is_stable:
    print("O sistema é ESTÁVEL, pois todos os polos estão dentro do círculo unitário.")
    if is_causal_if_stable: # Assumindo que queremos um sistema causal
        print("Assumindo uma Região de Convergência (ROC) causal (para fora do polo mais externo), o sistema também é CAUSAL.")
        print("Portanto, a função de transferência é ESTÁVEL e CAUSAL.")
else:
    print("O sistema NÃO é estável, pois um ou mais polos estão no ou fora do círculo unitário.")


# --- 5. Plotando Polos e Zeros no Plano Z ---
# Para plotar, precisamos dos zeros e polos como foram definidos (ou recalculados de num_poly_z, den_poly_z)
# z_plot, p_plot, k_plot = tf2zpk(num_poly_z, den_poly_z) # Reconfirmando a partir dos polinômios em z

print("\n--- Plotando Polos e Zeros no Plano Z ---")
fig, ax = plt.subplots(figsize=(7, 7))

# Plotar círculo unitário
unit_circle = plt.Circle((0, 0), 1, color='grey', fill=False, linestyle='--', linewidth=1)
ax.add_artist(unit_circle)

# Plotar zeros
if zeros.size > 0:
    ax.plot(np.real(zeros), np.imag(zeros), 'o', markersize=10, markerfacecolor='none', markeredgecolor='blue', label='Zeros')

# Plotar polos
if poles.size > 0:
    ax.plot(np.real(poles), np.imag(poles), 'x', markersize=10, markeredgecolor='red', label='Polos')

ax.set_xlabel("Parte Real ($\mathbb{R}$)")
ax.set_ylabel("Parte Imaginária ($\mathbb{I}$)")
ax.set_title("Plano Z: Polos e Zeros de $H(z)$ para ST9")
ax.grid(True, linestyle=':', linewidth=0.5)
ax.axhline(0, color='black', lw=0.5)
ax.axvline(0, color='black', lw=0.5)
ax.axis('equal')

# Definir limites do gráfico
all_coords = np.concatenate((np.real(zeros), np.real(poles), np.imag(zeros), np.imag(poles), [-1.1, 1.1]))
max_abs_val = np.max(np.abs(all_coords)) if all_coords.size > 0 else 1.1
plot_limit = np.ceil(max_abs_val * 1.5) # Adiciona uma margem
if plot_limit < 1.2:
    plot_limit = 1.2
ax.set_xlim([-plot_limit, plot_limit])
ax.set_ylim([-plot_limit, plot_limit])

ax.legend()
plt.show()