"""
NARMA10 物理リザバー評価
========================
Arduino(narma10_reservoir.ino)と通信して物理リザバーの状態を集め,
ソフトESNベースラインと *完全に同じ条件* で NRMSE を出す.

過去(0.7592)との違い:
  - データ長・ウォッシュアウト・NRMSE定義・train/test分割を
    narma10_esn_baseline.py と共通化 -> 初めてフェアな比較ができる

依存: numpy, pyserial   (pip install pyserial)
実行: python narma10_reservoir_eval.py

--- 設定 ---
"""
import time
import numpy as np
import serial

PORT = "COM4"          # 環境に合わせて変更 (Leonardoのポート)
BAUD = 115200
N_STEPS = 4000         # ベースラインと同じ
WASHOUT = 200          # ベースラインと同じ (過去は10で不十分だった)
LAM = 1e-6             # リッジ正則化 (ベースラインと同じ)
SEED_TASK = 0          # NARMA10生成seed (ベースラインと同じ)


# ---------------------------------------------------------------------------
# NARMA10 / NRMSE  (全スクリプト共通・改変禁止)
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


# ---------------------------------------------------------------------------
# 物理リザバーから状態収集
# ---------------------------------------------------------------------------
def collect_states(u):
    """入力系列 u を Arduino に流し, 各ステップの特徴ベクトルを集める."""
    pwm = (u * 2 * 255).clip(0, 255).astype(int)  # u:0-0.5 -> pwm:0-255

    print(f"接続中: {PORT} @ {BAUD}")
    ser = serial.Serial(PORT, BAUD, timeout=2)
    time.sleep(2.0)  # Leonardo再起動待ち
    ser.reset_input_buffer()

    states = []
    dim = None
    t0 = time.time()
    for i, p in enumerate(pwm):
        ser.write(f"{p}\n".encode())
        line = ser.readline().decode(errors="ignore").strip()
        try:
            vals = [int(v) for v in line.split(",")]
        except ValueError:
            vals = []
        if dim is None and vals:
            dim = len(vals)
            print(f"特徴次元 = {dim} (4ノード x {dim // 4}タイミング)")
        if dim and len(vals) == dim:
            states.append(vals)
        else:
            # 受信失敗時は直前の状態を複製(欠損対策)
            states.append(states[-1] if states else [0] * (dim or 16))

        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{len(pwm)}  ({rate:.0f} steps/s)")

    ser.close()
    X = np.array(states, dtype=float)
    # ADC 0-1023 を 0-1 に正規化(数値安定のため)
    return X / 1023.0


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------
def evaluate(X, y, washout=WASHOUT, lam=LAM):
    Xw, yw = X[washout:], y[washout:]
    split = len(Xw) // 2
    Wout = ridge_fit(Xw[:split], yw[:split], lam)
    pred = ridge_predict(Xw[split:], Wout)
    return nrmse(yw[split:], pred), yw[split:], pred


def main():
    print("NARMA10 生成中...")
    u, y = make_narma10(N_STEPS, seed=SEED_TASK)

    X = collect_states(u)
    print(f"収集完了: X shape = {X.shape}")
    np.savetxt("reservoir_states.csv", X, delimiter=",")   # ← この行を追加

    # --- 物理リザバーの評価 ---
    score, y_true, y_pred = evaluate(X, y)

    # --- 参考: 線形ベースライン(同じ入力・同じ条件) ---
    L = 11
    Xlin = np.zeros((N_STEPS - L, L))
    for i in range(L):
        Xlin[:, i] = u[L - 1 - i : N_STEPS - 1 - i]
    ylin = y[L:]
    s = len(Xlin) // 2
    Wl = ridge_fit(Xlin[:s], ylin[:s], LAM)
    lin = nrmse(ylin[s:], ridge_predict(Xlin[s:], Wl))

    print("\n" + "=" * 50)
    print(f"  物理リザバー NRMSE = {score:.4f}")
    print(f"  線形ベースライン   = {lin:.4f}  (これを下回れば第一関門突破)")
    print(f"  参考: ソフトESN    ≈ 0.55")
    print(f"  参考: 過去の記録   = 0.7592 (測定条件が違うため参考値)")
    print("=" * 50)
    if score < lin:
        print("  → 物理リザバーが線形を上回った. 非線形+メモリが効いている.")
    else:
        print("  → まだ線形以下. 観測タイミングやリカレント強度の調整余地あり.")

    # 予測波形をファイルに保存(matplotlibなしでも後で見られる)
    np.savetxt("narma10_result.csv",
               np.column_stack([y_true, y_pred]),
               delimiter=",", header="target,prediction", comments="")
    print("  予測結果を narma10_result.csv に保存.")


if __name__ == "__main__":
    main()