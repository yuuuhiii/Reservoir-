import numpy as np
import matplotlib.pyplot as plt

def lennard_jones_potential(r, epsilon, sigma):
    """
    レナード・ジョーンズポテンシャルを計算する関数
    """
    # (sigma/r)^12 が斥力項、(sigma/r)^6 が引力項
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

# パラメータの設定
epsilon = 1.0  # ポテンシャルの深さ
sigma = 1.0    # ポテンシャルが0になる距離

# 距離rの配列を生成 (0割りを防ぐため0より大きい値から開始)
# r が小さすぎると値が無限大に発散するため、0.8 * sigma から開始します
r = np.linspace(0.8 * sigma, 3.0 * sigma, 500)

# ポテンシャルの計算
V = lennard_jones_potential(r, epsilon, sigma)

# グラフの描画
plt.figure(figsize=(8, 6))
plt.plot(r, V, label='Lennard-Jones Potential', color='blue', linewidth=2)

# 基準線や特徴的な位置の描画
plt.axhline(0, color='black', linewidth=1, linestyle='--')  # V=0 の水平線
plt.axvline(sigma, color='gray', linewidth=1, linestyle=':', label='$\sigma$ (V=0)')
r_min = 2**(1/6) * sigma  # ポテンシャルが最小となる距離
plt.axvline(r_min, color='red', linewidth=1, linestyle=':', label=f'$r_{{min}}$ (Minimum)')

# グラフの表示範囲とラベルの設定
plt.ylim(-1.5 * epsilon, 2.0 * epsilon)
plt.xlim(0.8 * sigma, 3.0 * sigma)
plt.xlabel('Distance $r$')
plt.ylabel('Potential $V(r)$')
plt.title('Lennard-Jones Potential')
plt.legend()
plt.grid(True, alpha=0.5)

# グラフの表示
plt.show()