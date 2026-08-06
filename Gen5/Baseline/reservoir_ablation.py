"""
差分切り分けシミュレータ
========================
参照点:
  ESN(ソフト)      = 0.55  <- 動く
  RC過渡モデル(前) = 0.81  <- 動かない, 線形0.49に大敗

ESN にあって RC過渡モデルに無かったもの:
  (1) リカレント結合 W  : ノード出力が他ノードの入力に戻る -> 異時刻入力の混合
  (2) 入力重み Win の多様性 : ノードごとに異なる係数で入力を受ける

この2つを段階的に足し, どちらがスコアを動かすかを切り分ける.
効いた要素 = 物理回路に本当に必要な部品.

依存: numpy のみ
実行: python reservoir_ablation.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# NARMA10 / NRMSE (全スクリプト共通定義)
# ---------------------------------------------------------------------------
def make_narma10(n_steps, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=n_steps)
    y = np.zeros(n_steps)
    for t in range(n_steps - 1):
        if t < 10:
            y[t + 1] = 0.3 * y[t] + 0.1
        else:
            w = np.sum(y[t - 9 : t + 1])
            y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * w + 1.5 * u[t - 9] * u[t] + 0.1
    return u, y


def ridge_fit(X, y, lam=1e-6):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)


def ridge_predict(X, Wout):
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ Wout


def nrmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2) / np.var(a))


def evaluate_states(X, y, washout=200):
    Xw, yw = X[washout:], y[washout:]
    s = len(Xw) // 2
    Wout = ridge_fit(Xw[:s], yw[:s])
    return nrmse(yw[s:], ridge_predict(Xw[s:], Wout))


# ---------------------------------------------------------------------------
# 段階的にリッチにできる離散時間リザバーモデル
# ---------------------------------------------------------------------------
# 状態更新(離散, 1入力=1ステップ. 過渡の細分は今回は本質でないので簡略化):
#
#   x(t) = (1-a).*x(t-1) + a .* f( Win*u(t) + gamma * W @ x(t-1) )
#
#   - a       : リーク率 (RC時定数に対応). 段ごとに変えれば多様なメモリ.
#   - Win     : 入力重みベクトル. recurrent=False相当のとき全ノード同一.
#   - W       : リカレント結合行列. gamma=0 でフィードバック無効(前モデル相当).
#   - f       : 非線形 (tanh飽和 + 下側クリッピング)
#
# この1つの関数の引数を切り替えるだけで, 前モデル〜フルESN を連続的に再現する.

N_NODES = 4  # 物理ノード数の想定(過去は6). まず4で差分を見る.


def nonlinear(v):
    out = np.tanh(2.5 * v)
    return np.where(out < -0.2, -0.2, out)


def run_reservoir(u, leaks, Win, W, gamma, seed=1):
    """状態履歴 X(T, n) を返す."""
    T = len(u)
    n = len(leaks)
    X = np.zeros((T, n))
    x = np.zeros(n)
    for t in range(T):
        pre = Win * u[t] + gamma * (W @ x)
        x = (1 - leaks) * x + leaks * nonlinear(pre)
        X[t] = x
    return X


def build_config(n, diverse_win, recurrent, seed=1):
    """
    実験条件を組み立てる.
      diverse_win : True なら Win を段ごとに変える. False なら全段1.0(前モデル相当).
      recurrent   : True ならリカレント結合を有効化, False なら gamma=0.
    """
    rng = np.random.default_rng(seed)

    # リーク率: RC時定数のばらつきに対応(0.3ms〜5msを 0..1 に写像した感じ)
    leaks = np.array([0.8, 0.5, 0.25, 0.1])[:n]

    if diverse_win:
        Win = rng.uniform(0.3, 1.0, size=n)
    else:
        Win = np.ones(n)

    if recurrent:
        W = rng.uniform(-1, 1, size=(n, n))
        # スペクトル半径を0.9に正規化(エコーステート性の目安)
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        W = W * (0.9 / radius)
        gamma = 1.0
    else:
        W = np.zeros((n, n))
        gamma = 0.0

    return leaks, Win, W, gamma


# ---------------------------------------------------------------------------
# 実験本体
# ---------------------------------------------------------------------------
def main():
    N = 4000
    u, y = make_narma10(N, seed=0)

    # 線形ベースライン(第一関門)
    L = 11
    Xlin = np.zeros((N - L, L))
    for i in range(L):
        Xlin[:, i] = u[L - 1 - i : N - 1 - i]
    ylin = y[L:]
    s = len(Xlin) // 2
    Wout = ridge_fit(Xlin[:s], ylin[:s])
    lin = nrmse(ylin[s:], ridge_predict(Xlin[s:], Wout))

    print("=" * 60)
    print(" 差分切り分け: 何がNARMA10スコアを動かすか")
    print("=" * 60)
    print(f"  参照: 線形ベースライン = {lin:.4f}  (これを割れれば第一関門突破)")
    print(f"  参照: ソフトESN(n=200) = 0.55 前後")
    print(f"  参照: 前回RC過渡モデル  = 0.81 (動かなかった)")
    print(f"  ノード数 n = {N_NODES}")
    print("-" * 60)

    conditions = [
        ("(0) 前モデル相当   : Win一様, 帰還なし", False, False),
        ("(1) Win多様化のみ  : Win多様, 帰還なし", True,  False),
        ("(2) リカレントのみ : Win一様, 帰還あり", False, True),
        ("(3) 両方           : Win多様, 帰還あり", True,  True),
    ]

    results = {}
    for label, dwin, rec in conditions:
        # seedを複数振ってばらつきも見る(リカレントはseed依存が出るため)
        scores = []
        for sd in range(5):
            leaks, Win, W, gamma = build_config(N_NODES, dwin, rec, seed=sd)
            X = run_reservoir(u, leaks, Win, W, gamma)
            scores.append(evaluate_states(X, y))
        scores = np.array(scores)
        results[label] = scores
        print(f"{label}")
        print(f"       NRMSE = {scores.mean():.4f}  (best {scores.min():.4f}, "
              f"±{scores.std():.3f}, seed5回)")

    print("-" * 60)

    # ノード数を増やすとリカレント効果が伸びるか(物理ノードを何個作るべきか)
    print(" リカレント有効時, ノード数を増やすと?")
    for n in (4, 6, 8, 12, 20):
        scores = []
        for sd in range(5):
            leaks = np.linspace(0.85, 0.08, n)  # 時定数を連続的にばらつかせる
            rng = np.random.default_rng(sd)
            Win = rng.uniform(0.3, 1.0, size=n)
            W = rng.uniform(-1, 1, size=(n, n))
            W *= 0.9 / np.max(np.abs(np.linalg.eigvals(W)))
            X = run_reservoir(u, leaks, Win, W, 1.0)
            scores.append(evaluate_states(X, y))
        scores = np.array(scores)
        print(f"       n={n:2d}  NRMSE = {scores.mean():.4f}  (best {scores.min():.4f})")

    print("=" * 60)
    print(" 読み方:")
    print("  ・(2)や(3)が(0)より大きく下がる = リカレント結合が本質")
    print("    -> 物理は『ノード電圧を別ノード入力へ戻す配線』を最優先")
    print("  ・(1)だけで下がる = 入力重みの多様性が本質")
    print("    -> 物理は『各ノードに異なる入力分圧』を用意")
    print("  ・ノード数スキャンで頭打ちになる点 = 作るべき物理ノード数の目安")
    print("=" * 60)


if __name__ == "__main__":
    main()