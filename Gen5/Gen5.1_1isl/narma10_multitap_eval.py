"""
NARMA10 連続駆動 + 複数タイミング観測 評価
============================================
narma10_multitap.ino と組む. 受信列数は可変(自動検出).
4ノード x Nタイミングで次元を増やし, 実効次元不足を解消できるか見る.

依存: numpy, pyserial
実行: python narma10_multitap_eval.py
"""
import time
import numpy as np
import serial

PORT = "COM4"
BAUD = 115200
N_STEPS = 4000
WASHOUT = 200
LAM = 1e-6
SEED_TASK = 0
BATCH = 64


def make_narma10(n, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 0.5, size=n); y = np.zeros(n)
    for t in range(n - 1):
        if t < 10:
            y[t + 1] = 0.3 * y[t] + 0.1
        else:
            w = np.sum(y[t - 9 : t + 1])
            y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * w + 1.5 * u[t - 9] * u[t] + 0.1
    return u, y


def ridge_fit(X, y, lam=1e-6):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)

def ridge_predict(X, W): return np.hstack([X, np.ones((X.shape[0], 1))]) @ W
def nrmse(a, b): return np.sqrt(np.mean((a - b) ** 2) / np.var(a))


def collect_states(u):
    pwm = (u * 2 * 255).clip(0, 255).astype(int)
    print(f"接続中: {PORT} @ {BAUD}")
    ser = serial.Serial(PORT, BAUD, timeout=3)
    time.sleep(2.0)
    ser.reset_input_buffer()

    states = []; dim = None; t0 = time.time()
    for b0 in range(0, len(pwm), BATCH):
        batch = pwm[b0 : b0 + BATCH]
        ser.write((",".join(str(int(p)) for p in batch) + "\n").encode())
        for _ in range(len(batch)):
            line = ser.readline().decode(errors="ignore").strip()
            try:
                vals = [int(v) for v in line.split(",")]
            except ValueError:
                vals = []
            if dim is None and vals:
                dim = len(vals)
                print(f"特徴次元 = {dim} (4ノード x {dim//4}タイミング)")
            if dim and len(vals) == dim:
                states.append(vals)
            else:
                states.append(states[-1] if states else [0] * (dim or 12))
        if (b0 // BATCH) % 10 == 0 and b0 > 0:
            print(f"  {len(states)}/{len(pwm)}  ({len(states)/(time.time()-t0):.0f} steps/s)")
    ser.close()
    return np.array(states, dtype=float) / 1023.0


def main():
    u, y = make_narma10(N_STEPS, seed=SEED_TASK)
    X = collect_states(u)
    print(f"収集完了: X shape = {X.shape}")
    np.savetxt("reservoir_states.csv", X, delimiter=",")

    Xw, yw = X[WASHOUT:], y[WASHOUT:]
    s = len(Xw) // 2
    Wout = ridge_fit(Xw[:s], yw[:s], LAM)
    score = nrmse(yw[s:], ridge_predict(Xw[s:], Wout))

    L = 11
    Xl = np.zeros((N_STEPS - L, L))
    for i in range(L):
        Xl[:, i] = u[L - 1 - i : N_STEPS - 1 - i]
    yl = y[L:]; sl = len(Xl) // 2
    Wl = ridge_fit(Xl[:sl], yl[:sl], LAM)
    lin = nrmse(yl[sl:], ridge_predict(Xl[sl:], Wl))

    # 実効次元(特異値)
    sv = np.linalg.svd(X - X.mean(0), compute_uv=False)
    eff = (sv.sum() ** 2) / (sv ** 2).sum()  # participation ratio的な目安

    print("\n" + "=" * 50)
    print(f"  物理リザバー NRMSE = {score:.4f}")
    print(f"  線形ベースライン   = {lin:.4f}")
    print(f"  実効次元(目安)     = {eff:.1f} / {X.shape[1]}")
    print("=" * 50)
    if score < lin:
        print("  → 線形突破.")

    np.savetxt("narma10_result.csv",
               np.column_stack([yw[s:], ridge_predict(Xw[s:], Wout)]),
               delimiter=",", header="target,prediction", comments="")


if __name__ == "__main__":
    main()
