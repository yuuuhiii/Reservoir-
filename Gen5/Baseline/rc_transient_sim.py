"""
RC transient reservoir simulator
=================================
検証したい仮説:
  「過去回路の不振は素子不足ではなく delay(5) による定常観測が原因.
   観測タイミングを変えるだけ(=過渡を読む/複数タイミングで読む)で
   NARMA10 スコアは改善するのではないか.」

このスクリプトは Arduino 不要. 過去回路を「非線形素子 + 複数RC段」として
粗くモデル化し, 観測タイミング戦略を3つ比較する:

  戦略A: delay(5ms) 相当 = 定常状態を1回読む (過去システム)
  戦略B: delay(0.5ms) 相当 = 過渡の途中を1回読む
  戦略C: time-multiplex = 1入力に対し複数タイミングで読む(物理ノードは増やさない)

依存: numpy のみ
実行: python rc_transient_sim.py

注意: これは「相対比較」のための粗いモデル.
      絶対NRMSEを物理と一致させる装置ではなく,
      「タイミングを変えると相対的にどう動くか」の傾向を見るためのもの.
"""

import numpy as np


# ---------------------------------------------------------------------------
# NARMA10 と NRMSE (ソフトESN版と同一定義・同一条件で揃える)
# ---------------------------------------------------------------------------
def make_narma10(n_steps, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=n_steps)
    y = np.zeros(n_steps)
    for t in range(n_steps - 1):
        if t < 10:
            y[t + 1] = 0.3 * y[t] + 0.1
        else:
            window = np.sum(y[t - 9 : t + 1])
            y[t + 1] = (
                0.3 * y[t] + 0.05 * y[t] * window + 1.5 * u[t - 9] * u[t] + 0.1
            )
    return u, y


def ridge_fit(X, y, lam=1e-6):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ y)


def ridge_predict(X, Wout):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return Xb @ Wout


def nrmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2) / np.var(y_true))


def evaluate_states(X, y, washout=200):
    """状態履歴 X(T,d) と教師 y から NRMSE を出す. 条件はソフトESN版と共通."""
    Xw, yw = X[washout:], y[washout:]
    split = len(Xw) // 2
    Wout = ridge_fit(Xw[:split], yw[:split])
    pred = ridge_predict(Xw[split:], Wout)
    return nrmse(yw[split:], pred)


# ---------------------------------------------------------------------------
# 物理回路の粗いモデル
# ---------------------------------------------------------------------------
# 過去回路の要素:
#   - PWM を平滑化したアナログ入力 (入力電圧)
#   - 非線形素子 (MOSFETの飽和 = tanh的, ダイオードのクリッピング)
#   - 複数のRC段 (異なる時定数のフェーディングメモリ)
#
# 連続時間の微分方程式を細かい時間刻み dt でオイラー積分し,
# 「1入力ステップ = STEP_DURATION の実時間」の中を高分解能でシミュレートする.
# こうすることで「入力から t 秒後に読む」という観測タイミングを表現できる.

STEP_DURATION = 5.0e-3   # 1入力ステップの実時間 [s] (過去の delay(5) に対応)
DT = 5.0e-5              # 内部積分の時間刻み [s] (STEP_DURATION を100分割)
N_SUBSTEPS = int(STEP_DURATION / DT)

# 複数RC段の時定数 [s]. わざとバラつかせて多様なメモリを持たせる.
# 過去回路の「1000µF のダム」から小容量まで混在していた状況を模す.
TAU_LIST = [0.3e-3, 0.8e-3, 2.0e-3, 5.0e-3]   # 0.3ms 〜 5ms


def nonlinear(v):
    """非線形素子: MOSFET飽和(tanh) + ダイオードclipping(負を浅く)."""
    out = np.tanh(2.5 * v)          # 飽和
    out = np.where(out < -0.2, -0.2, out)  # 下側クリッピング
    return out


def simulate_reservoir(u, sample_times):
    """
    入力系列 u を回路に流し, 各入力ステップについて sample_times で指定した
    複数タイミング(ステップ開始からの経過割合 0.0〜1.0)で全RC段の電圧を読む.

    戻り値 X: shape (T, len(TAU_LIST) * len(sample_times))
             = 各時定数 × 各観測タイミング を特徴次元として並べたもの.
    """
    T = len(u)
    n_tau = len(TAU_LIST)
    n_samp = len(sample_times)
    X = np.zeros((T, n_tau * n_samp))

    # 各RC段の状態電圧
    v = np.zeros(n_tau)

    # sample_times(割合) を substep index に変換
    samp_idx = [min(int(s * N_SUBSTEPS), N_SUBSTEPS - 1) for s in sample_times]

    for t in range(T):
        drive = nonlinear(u[t])   # 非線形素子を通した入力
        captured = np.zeros((n_samp, n_tau))
        si = 0
        for sub in range(N_SUBSTEPS):
            # 各RC段: dv/dt = (drive - v) / tau
            for k, tau in enumerate(TAU_LIST):
                v[k] += DT * (drive - v[k]) / tau
            # このサブステップが観測タイミングなら記録
            if si < n_samp and sub == samp_idx[si]:
                captured[si] = v.copy()
                si += 1
        X[t] = captured.flatten()
    return X


# ---------------------------------------------------------------------------
# 3戦略の比較
# ---------------------------------------------------------------------------
def main():
    N = 4000
    u, y = make_narma10(N, seed=0)

    print("=" * 56)
    print(" RC過渡リザバー: 観測タイミング戦略の比較 (NARMA10)")
    print("=" * 56)
    print(f"  1ステップ実時間 = {STEP_DURATION*1e3:.1f} ms, 内部刻み {DT*1e6:.0f} us")
    print(f"  RC時定数 = {[f'{t*1e3:.1f}ms' for t in TAU_LIST]}")
    print()

    # --- 参考: 線形ベースライン(リザバーなし, 入力遅延ベクトル) ---
    L = 11
    Xlin = np.zeros((N - L, L))
    for i in range(L):
        Xlin[:, i] = u[L - 1 - i : N - 1 - i]
    ylin = y[L:]
    print(f"[参考] 線形ベースライン NRMSE = {evaluate_states(np.column_stack([Xlin]), np.concatenate([[0]*0, ylin]), washout=200):.4f}")
    # ↑ washout込みで揃えるため簡易に評価
    split = len(Xlin) // 2
    Wout = ridge_fit(Xlin[:split], ylin[:split])
    lin = nrmse(ylin[split:], ridge_predict(Xlin[split:], Wout))
    print(f"        (この値を物理が下回れるかが第一関門) 再掲: {lin:.4f}\n")

    # --- 戦略A: 定常観測 (過去システム, delay(5ms)相当) ---
    # ステップ終端(割合1.0)で1回だけ読む = 過渡が落ち着いた後
    Xa = simulate_reservoir(u, sample_times=[1.0])
    print(f"戦略A  定常1点観測  (delay 5ms 相当, 過去システム)")
    print(f"        次元 = {Xa.shape[1]:2d}   NRMSE = {evaluate_states(Xa, y):.4f}\n")

    # --- 戦略B: 過渡観測 (delay(0.5ms)相当) ---
    # ステップの序盤(割合0.1)で1回読む = 過渡の途中
    Xb = simulate_reservoir(u, sample_times=[0.1])
    print(f"戦略B  過渡1点観測  (delay 0.5ms 相当)")
    print(f"        次元 = {Xb.shape[1]:2d}   NRMSE = {evaluate_states(Xb, y):.4f}\n")

    # --- 戦略C: 時間多重 (1入力に対し複数タイミングで読む) ---
    for samp in ([0.1, 0.3, 0.6, 1.0],
                 [0.05, 0.15, 0.3, 0.5, 0.7, 1.0],
                 [0.02, 0.06, 0.12, 0.2, 0.32, 0.5, 0.7, 1.0]):
        Xc = simulate_reservoir(u, sample_times=samp)
        print(f"戦略C  時間多重 {len(samp)}タイミング  "
              f"次元 = {Xc.shape[1]:2d}   NRMSE = {evaluate_states(Xc, y):.4f}")

    print()
    print("=" * 56)
    print(" 読み方:")
    print("  ・A(定常) より B(過渡) が良ければ = delay(5)が記憶を消していた証拠")
    print("  ・C(時間多重) が線形を明確に下回れば = 配線を増やさず勝てる証拠")
    print("    → その場合, 回路はほぼ現状維持でスケッチのタイミング制御だけ変更")
    print("=" * 56)


if __name__ == "__main__":
    main()