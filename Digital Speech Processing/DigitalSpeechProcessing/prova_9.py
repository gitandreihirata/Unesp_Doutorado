import numpy as np

pi = np.array([0.7, 0.2, 0.1])
A = np.array([[0.6, 0.3, 0.1], [0.4, 0.1, 0.5], [0.1, 0.7, 0.2]])
B = np.array([[0.1, 0.8], [0.3, 0.3], [0.6, 0.1]])

obs_map = {'rainy': 0, 'sunny': 1}
O = ['rainy', 'sunny']

obs_seq = [obs_map[ob] for ob in O]
N = A.shape[0]
T = len(obs_seq)

alpha = np.zeros((T, N))
alpha[0, :] = pi * B[:, obs_seq[0]]

for t in range(1, T):
    for j in range(N):
        alpha[t, j] = np.sum(alpha[t-1, :] * A[:, j]) * B[j, obs_seq[t]]

prob_forward = np.sum(alpha[T-1, :])

delta = np.zeros((T, N))
psi = np.zeros((T, N), dtype=int)

delta[0, :] = pi * B[:, obs_seq[0]]

for t in range(1, T):
    for j in range(N):
        probs = delta[t-1, :] * A[:, j]
        delta[t, j] = np.max(probs) * B[j, obs_seq[t]]
        psi[t, j] = np.argmax(probs)

prob_viterbi = np.max(delta[T-1, :])
best_path = np.zeros(T, dtype=int)
best_path[T-1] = np.argmax(delta[T-1, :])

for t in range(T-2, -1, -1):
    best_path[t] = psi[t+1, best_path[t+1]]

print("--- Prova Final: Exercício (9) ---")
print("\n1. Probabilidade P(O|λ) (Forward Algorithm):")
print(f"{prob_forward:.4f}")
print("\n2. Melhor Caminho de Estados (Viterbi Algorithm):")
print(f"Caminho: {[f'q{state}' for state in best_path]}")
print(f"Probabilidade do melhor caminho: {prob_viterbi:.4f}")