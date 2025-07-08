import numpy as np

y = np.array([2, 5, 2, 3, 5, 8, 4, 8, 10, 6])
p = 4
N = len(y)

num_equations = N - p
X = np.zeros((num_equations, p))
yp = y[p:]

for i in range(num_equations):
    X[i, :] = y[i : i+p]

XTX = X.T @ X
XTy = X.T @ yp

a_coeffs_reversed = np.linalg.solve(XTX, XTy)

print("--- Prova Final: Exercício (7) ---")
print("\nCoeficientes LPC de 4ª ordem:")
for i, coeff in enumerate(a_coeffs_reversed):
    print(f"a_{i+1} = {coeff:.4f}")