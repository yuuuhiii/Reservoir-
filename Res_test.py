import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

t = np.linspace(0, 20, 500)
u = np.sin(t) 
target = np.roll(u, -1)

np.random.seed(42)
W_in = np.random.rand(100, 1) * 2 - 1
W_res = np.random.rand(100, 100) * 2 - 1

radius = np.max(np.abs(np.linalg.eigvals(W_res)))
W_res = W_res * (0.9 / radius)

X = np.zeros((len(t), 100))
x_current = np.zeros(100)

for i in range(len(t)):
    x_current = np.tanh(np.dot(W_in, [u[i]]) + np.dot(W_res, x_current))
    X[i, :] = x_current

X_train = X[:-1]
target_train = target[:-1]
ridge = Ridge(alpha=1e-4)
ridge.fit(X_train, target_train)      
predictions = ridge.predict(X_train)  

# グラフの描画
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(t[:-1], u[:-1], color="black", linewidth=2)
plt.title("Step 1: Input Signal $u(t)$")
plt.grid(True, alpha=0.3)
plt.subplot(3, 1, 2)
for j in range(5):
    plt.plot(t[:-1], X_train[:, j], alpha=0.8)
plt.title("Step 2: Intermediate States $\mathbf{x}(t)$ (Inside the Reservoir)")
plt.grid(True, alpha=0.3)
plt.subplot(3, 1, 3)
plt.plot(t[:-1], target_train, label='True Future Value', linestyle='--', color='gray', linewidth=2)
plt.plot(t[:-1], predictions, label='Reservoir Output (Prediction)', color='red', alpha=0.7, linewidth=2)
plt.title("Step 3: Final Output $y(t)$ (Result)")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()