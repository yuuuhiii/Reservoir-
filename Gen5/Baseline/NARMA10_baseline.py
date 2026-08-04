"""
NARMA10 baseline with a software Echo State Network (ESN).

目的:
  物理リザバーを作る前に「NARMA10でどのくらいのNRMSEが狙えるか」の相場を出す.
  読み出し(リッジ回帰)・NRMSE計算・NARMA10生成のコードはこの後の物理版と共通で使える.

依存: numpy のみ
実行: python narma10_esn_baseline.py

用語:
  NRMSE = 正規化二乗平均平方根誤差. 小さいほど良い. 分散で正規化しているので
          「1.0 = 平均値を出すだけのモデルと同等」が目安. 0.7592は「平均より少しマシ」寄り.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. NARMA10 タスクの生成
# ---------------------------------------------------------------------------
def make_narma10(n_steps, seed=0):
    """
    NARMA10: 過去10ステップの出力履歴と入力に依存する非線形自己回帰タスク.
    入力 u(t) は [0, 0.5] の一様乱数. 教師 y(t) は下の漸化式で定義される.

    y(t+1) = 0.3 y(t) + 0.05 y(t) * (sum_{i=0..9} y(t-i)) + 1.5 u(t-9) u(t) + 0.1
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=n_steps)
    y = np.zeros(n_steps)
    for t in range(n_steps - 1):
        if t < 10:
            # 立ち上がり区間は履歴が不足するので簡易更新(ウォッシュアウトで捨てる)
            y[t + 1] = 0.3 * y[t] + 0.1
        else:
            window = np.sum(y[t - 9 : t + 1])
            y[t + 1] = (
                0.3 * y[t]
                + 0.05 * y[t] * window
                + 1.5 * u[t - 9] * u[t]
                + 0.1
            )
    return u, y


# ---------------------------------------------------------------------------
# 2. Echo State Network (リザバー本体)
# ---------------------------------------------------------------------------
class ESN:
    """
    標準的な leaky-integrator ESN.
      x(t) = (1-a) x(t-1) + a * tanh(Win * u(t) + W * x(t-1))
    W はスペクトル半径 rho に正規化する(エコーステート性の目安).
    """

    def __init__(self, n_reservoir=200, spectral_radius=0.9, leak=0.3,
                 input_scale=0.5, sparsity=0.1, seed=1):
        self.n = n_reservoir
        self.a = leak
        rng = np.random.default_rng(seed)

        # 入力重み
        self.Win = rng.uniform(-1, 1, size=(n_reservoir, 1)) * input_scale

        # リカレント重み(疎行列 -> スペクトル半径で正規化)
        W = rng.uniform(-1, 1, size=(n_reservoir, n_reservoir))
        mask = rng.uniform(0, 1, size=W.shape) > sparsity
        W[mask] = 0.0
        eigs = np.linalg.eigvals(W)
        radius = np.max(np.abs(eigs))
        self.W = W * (spectral_radius / radius)

    def run(self, u):
        """入力系列 u を流してリザバー状態の履歴 X (T, n) を返す."""
        T = len(u)
        X = np.zeros((T, self.n))
        x = np.zeros(self.n)
        for t in range(T):
            pre = self.Win[:, 0] * u[t] + self.W @ x
            x = (1 - self.a) * x + self.a * np.tanh(pre)
            X[t] = x
        return X


# ---------------------------------------------------------------------------
# 3. リッジ回帰による読み出し + NRMSE  (物理版と共通で使う部分)
# ---------------------------------------------------------------------------
def ridge_fit(X, y, lam=1e-6):
    """読み出し重み Wout を閉形式で解く. X にバイアス列を足す."""
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    b = Xb.T @ y
    return np.linalg.solve(A, b)


def ridge_predict(X, Wout):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return Xb @ Wout


def nrmse(y_true, y_pred):
    """分散正規化RMSE. 1.0 = 平均値予測と同等, 0 = 完全一致."""
    mse = np.mean((y_true - y_pred) ** 2)
    var = np.var(y_true)
    return np.sqrt(mse / var)


# ---------------------------------------------------------------------------
# 4. 評価パイプライン
# ---------------------------------------------------------------------------
def evaluate(n_reservoir=200, spectral_radius=0.9, leak=0.3, input_scale=0.5,
             lam=1e-6, n_steps=4000, washout=200, seed_task=0, seed_esn=1):
    u, y = make_narma10(n_steps, seed=seed_task)
    esn = ESN(n_reservoir, spectral_radius, leak, input_scale, seed=seed_esn)
    X = esn.run(u)

    # ウォッシュアウト後を train/test に分割
    Xw, yw = X[washout:], y[washout:]
    split = len(Xw) // 2
    Xtr, ytr = Xw[:split], yw[:split]
    Xte, yte = Xw[split:], yw[split:]

    Wout = ridge_fit(Xtr, ytr, lam)
    pred = ridge_predict(Xte, Wout)
    return nrmse(yte, pred)


# ---------------------------------------------------------------------------
# 5. 実行: ベースライン + 簡単なパラメータ探索
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== 基準: 線形回帰のみ (リザバーなし, 入力の遅延ベクトルだけ) ===")
    # 物理リザバーの「下限ライン」. これを下回れないなら回路は無意味.
    u, y = make_narma10(4000, seed=0)
    L = 11
    Xlin = np.zeros((len(u) - L, L))
    for i in range(L):
        Xlin[:, i] = u[L - 1 - i : len(u) - 1 - i]
    ylin = y[L:]
    split = len(Xlin) // 2
    Wout = ridge_fit(Xlin[:split], ylin[:split], 1e-6)
    pred = ridge_predict(Xlin[split:], Wout)
    print(f"  線形ベースライン NRMSE = {nrmse(ylin[split:], pred):.4f}\n")

    print("=== ESN ベースライン (n=200) ===")
    print(f"  NRMSE = {evaluate():.4f}\n")

    print("=== パラメータ探索 (spectral_radius x leak) ===")
    for rho in (0.7, 0.9, 1.1):
        for leak in (0.1, 0.3, 0.6):
            score = evaluate(spectral_radius=rho, leak=leak)
            print(f"  rho={rho:.1f}  leak={leak:.1f}  ->  NRMSE = {score:.4f}")

    print("\n=== リザバーサイズ依存 ===")
    for n in (20, 50, 100, 200, 400):
        score = evaluate(n_reservoir=n)
        print(f"  n={n:4d}  ->  NRMSE = {score:.4f}")