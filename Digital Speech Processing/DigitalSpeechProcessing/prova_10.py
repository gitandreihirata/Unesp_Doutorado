import numpy as np

Ca = np.array([0.2, 1.0, 0.7])
Cb = np.array([0.3, 0.9, 0.8])
t = np.array([0.2, 0.9, 0.9])

dist_A = np.linalg.norm(t - Ca)
dist_B = np.linalg.norm(t - Cb)

best_class = 'A' if dist_A < dist_B else 'B'

print("--- Prova Final: Exercício (10) ---")
print("\nParte 1: Classificação por Distância Euclidiana")
print(f"Distância para a Classe A: {dist_A:.4f}")
print(f"Distância para a Classe B: {dist_B:.4f}")
print(f"Melhor correspondência: Classe {best_class}")