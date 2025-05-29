import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Coeficientes do filtro
b = [1, -2]      # Numerador: 1 - 2z⁻¹
a = [1, -0.5]    # Denominador: 1 - 0.5z⁻¹

# --- 1. Equação de Diferenças ---
# A função de transferência é H(z) = Y(z)/X(z)
# Y(z) * (a0 + a1*z^-1) = X(z) * (b0 + b1*z^-1)
# No domínio do tempo:
# a0*y[n] + a1*y[n-1] = b0*x[n] + b1*x[n-1]
# Assumindo a0 = 1 (se não, normalizar):
# y[n] = -a1*y[n-1] + b0*x[n] + b1*x[n-1]

# Para a0 = 1, a1 = -0.5, b0 = 1, b1 = -2:
# y[n] = -(-0.5)*y[n-1] + 1*x[n] + (-2)*x[n-1]
# y[n] = 0.5*y[n-1] + x[n] - 2*x[n-1]
# Equação da diferença
print("Equação da diferença:")
print("y[n] = 0.5*y[n-1] + x[n] - 2*x[n-1]")


zeros, poles, _ = signal.tf2zpk(b, a)

print("\nZeros:", zeros)
print("Polos:", poles)



