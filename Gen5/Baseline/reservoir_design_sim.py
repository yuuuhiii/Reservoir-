"""
汎用リザバー設計シミュレータ (物理を組む前の設計図づくり)
==========================================================
狙い:
  時定数を広く散らした6ノード + 複数リカレント結合が,
  性格の異なる3タスクを同時にどれだけ解けるか(=汎用性)を測る.
  最適構成を物理素子値(各ノードのRC, ノード間結合)に翻訳する材料にする.

3タスク:
  1) 短期記憶     : u(t-k) の遅延再現  -> 記憶の長さを測る
  2) 波形分類     : 矩形波/三角波の区別 -> 非線形分類能力(ピエゾ素材判定の布石)
  3) NARMA10      : 中記憶+非線形の混合 -> 包含目標

リザバーモデル (離散時間, 物理RCの近似):
  x(t) = (1-a).*x(t-1) + a .* f( Win*u(t) + Wrec @ x(t-1) )
  - a[i]      : ノードごとのリーク率(RC時定数に対応). 対数的に散らす.
  - Wrec      : リカレント結合行列(疎). gamma で強さ調整.
  - f         : tanh飽和 + 下側クリップ(MOSFET非線形の近似)

依存: numpy のみ
実行: python reservoir_design_sim.py
"""
import numpy as np


# ---------------------------------------------------------------------------
# 共通: 読み出しと評価指標
# ---------------------------------------------------------------------------
def ridge_fit(X, y, lam=1e-4):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)

def ridge_predict(X, W):
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ W

def nrmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2) / np.var(a))

def r2(a, b):
    return max(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2), 0)


# ---------------------------------------------------------------------------
# リザバー本体
# ---------------------------------------------------------------------------
def nonlinear(v):
    out = np.tanh(1.5 * v)
    return np.where(out < -0.2, -0.2, out)


def make_reservoir(n_nodes=6, tau_min_ms=1, tau_max_ms=300, step_ms=6,
                   n_recur=6, gamma=0.6, seed=1):
    """
    時定数を対数的に散らした n_nodes ノード + 疎なリカレント結合を構成.
    a[i] = step_ms / tau[i] を 0-1 にクリップ(1ステップでどれだけ更新されるか).
    """
    rng = np.random.default_rng(seed)
    taus = np.logspace(np.log10(tau_min_ms), np.log10(tau_max_ms), n_nodes)
    leaks = np.clip(step_ms / taus, 0.02, 1.0)

    Win = rng.uniform(0.5, 1.0, size=n_nodes) * rng.choice([-1, 1], n_nodes)

    # 疎なリカレント: n_recur 本だけ結合を張る
    Wrec = np.zeros((n_nodes, n_nodes))
    for _ in range(n_recur):
        i, j = rng.integers(0, n_nodes, 2)
        if i != j:
            Wrec[i, j] = rng.uniform(-1, 1)
    # スペクトル半径を gamma に正規化(暴走防止)
    radius = np.max(np.abs(np.linalg.eigvals(Wrec))) if n_recur > 0 else 0
    if radius > 1e-6:
        Wrec *= gamma / radius

    return {"leaks": leaks, "Win": Win, "Wrec": Wrec, "taus": taus}


def run(res, u):
    leaks, Win, Wrec = res["leaks"], res["Win"], res["Wrec"]
    n = len(leaks); T = len(u)
    X = np.zeros((T, n)); x = np.zeros(n)
    for t in range(T):
        pre = Win * u[t] + Wrec @ x
        x = (1 - leaks) * x + leaks * nonlinear(pre)
        X[t] = x
    return X


# ---------------------------------------------------------------------------
# 3タスク
# ---------------------------------------------------------------------------
def task_narma10(n, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, n); y = np.zeros(n)
    for t in range(n - 1):
        if t < 10: y[t+1] = 0.3*y[t] + 0.1
        else:
            w = np.sum(y[t-9:t+1]); y[t+1] = 0.3*y[t] + 0.05*y[t]*w + 1.5*u[t-9]*u[t] + 0.1
    return u, y

def task_memory(n, k, seed=1):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, n)
    y = np.concatenate([np.zeros(k), u[:-k]])  # u(t-k)
    return u, y

def task_waveform(n, seed=2):
    """矩形波/三角波をランダムに切り替え, 現在どちらかを分類(0/1)."""
    rng = np.random.default_rng(seed)
    u = np.zeros(n); y = np.zeros(n)
    period = 10; t = 0
    while t < n:
        kind = rng.integers(0, 2)
        length = period * rng.integers(2, 5)
        for i in range(length):
            if t >= n: break
            ph = (i % period) / period
            if kind == 0:  # 矩形波
                u[t] = 0.5 if ph < 0.5 else 0.0
            else:          # 三角波
                u[t] = ph if ph < 0.5 else (1 - ph)
            y[t] = kind
            t += 1
    return u, y


def eval_task(res, u, y, washout=200, classify=False):
    X = run(res, u)
    Xw, yw = X[washout:], y[washout:]
    s = len(Xw) // 2
    W = ridge_fit(Xw[:s], yw[:s])
    p = ridge_predict(Xw[s:], W)
    if classify:
        acc = np.mean((p > 0.5).astype(int) == yw[s:].astype(int))
        return acc  # 正解率(高いほど良い)
    return nrmse(yw[s:], p)  # NRMSE(低いほど良い)


# ---------------------------------------------------------------------------
# 実験
# ---------------------------------------------------------------------------
def evaluate_config(label, **kwargs):
    N = 4000
    scores = {"narma": [], "mem5": [], "wave": []}
    for seed in range(3):
        res = make_reservoir(seed=seed, **kwargs)
        u, y = task_narma10(N, seed=0); scores["narma"].append(eval_task(res, u, y))
        u, y = task_memory(N, k=5, seed=1); scores["mem5"].append(eval_task(res, u, y))
        u, y = task_waveform(N, seed=2); scores["wave"].append(eval_task(res, u, y, classify=True))
    print(f"{label}")
    print(f"    NARMA10 NRMSE = {np.mean(scores['narma']):.3f}  "
          f"| 記憶k=5 NRMSE = {np.mean(scores['mem5']):.3f}  "
          f"| 波形分類 acc = {np.mean(scores['wave']):.3f}")


def main():
    print("=" * 68)
    print(" 汎用リザバー設計: 時定数レンジとリカレントの効果 (6ノード)")
    print(" NARMA10/記憶: NRMSE(低いほど良) | 波形分類: 正解率(高いほど良)")
    print("=" * 68)

    print("\n--- 時定数レンジの効果 (リカレントなし) ---")
    evaluate_config(" 狭い時定数 1-10ms (今の回路に近い)",
                    tau_min_ms=1, tau_max_ms=10, n_recur=0)
    evaluate_config(" 中程度   1-60ms",
                    tau_min_ms=1, tau_max_ms=60, n_recur=0)
    evaluate_config(" 広い     1-300ms (対数分散)",
                    tau_min_ms=1, tau_max_ms=300, n_recur=0)

    print("\n--- リカレント強度の効果 (時定数1-300ms固定) ---")
    for g in (0.0, 0.3, 0.6, 0.9):
        evaluate_config(f" リカレント gamma={g}",
                        tau_min_ms=1, tau_max_ms=300, n_recur=6, gamma=g)

    print("\n--- ノード数の効果 (時定数1-300ms, gamma=0.6) ---")
    for nn in (4, 6, 8, 12):
        evaluate_config(f" {nn}ノード",
                        n_nodes=nn, tau_min_ms=1, tau_max_ms=300, n_recur=nn, gamma=0.6)

    print("\n" + "=" * 68)
    print(" 読み方:")
    print("  ・時定数を広げると3タスクが同時に底上げ = 汎用性の証拠")
    print("  ・gammaを上げてどこで暴れる(悪化する)かが物理リカレントの安全域")
    print("  ・最良構成のtaus(時定数)がそのまま各ノードのRC設計値になる")
    print("=" * 68)


if __name__ == "__main__":
    main()