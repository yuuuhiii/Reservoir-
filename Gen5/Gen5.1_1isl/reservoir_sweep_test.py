"""
切り分けスクリプト: 本番の駆動法で診断の綺麗な波形が出るか
==========================================================
本番スケッチ(narma10_reservoir.ino)をそのまま使い, Python側から流す入力を
「NARMA10のランダム」ではなく「診断と同じ三角波(0->255->0)」に差し替える.

判定:
  - 三角波で状態が 0-1023 の広い範囲でフルスイングする
    -> 駆動法・通信は健全. 問題はNARMA10入力の出し方(分布/レンジ)に限定
  - 三角波でも 0付近に張り付く
    -> 本番スケッチの駆動法そのもの(往復通信の遅さ等)が原因. 通信を作り直す

依存: numpy, pyserial
実行: python reservoir_sweep_test.py
"""
import time
import numpy as np
import serial

PORT = "COM4"
BAUD = 115200
N_STEPS = 1000   # 切り分けなので短くてよい


def main():
    # 診断と同じ三角波 (0->255->0 を繰り返す)
    ramp = np.concatenate([np.arange(0, 256, 4), np.arange(255, -1, -4)])
    pwm = np.tile(ramp, (N_STEPS // len(ramp)) + 1)[:N_STEPS].astype(int)

    print(f"接続中: {PORT} @ {BAUD}")
    ser = serial.Serial(PORT, BAUD, timeout=2)
    time.sleep(2.0)
    ser.reset_input_buffer()

    states = []
    dim = None
    for i, p in enumerate(pwm):
        ser.write(f"{int(p)}\n".encode())
        line = ser.readline().decode(errors="ignore").strip()
        try:
            vals = [int(v) for v in line.split(",")]
        except ValueError:
            vals = []
        if dim is None and vals:
            dim = len(vals)
        if dim and len(vals) == dim:
            states.append(vals)
        else:
            states.append(states[-1] if states else [0] * (dim or 16))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{N_STEPS}")
    ser.close()

    X = np.array(states, dtype=float)
    np.savetxt("sweep_states.csv", X, delimiter=",")

    # 各特徴のレンジを表示 (フルスイングしているか)
    nodes = ["N0(1ms)", "N1(10ms)", "N2(100ms)", "N3(470ms)"]
    times = ["1ms", "3ms", "6ms", "12ms"]
    labels = [f"{nodes[n]}@{times[k]}" for k in range(4) for n in range(4)]

    print("\n=== 三角波駆動での各特徴レンジ (生ADC 0-1023) ===")
    print(f'{"特徴":<20}{"min":>6}{"max":>6}{"range":>7}')
    for i in range(16):
        xi = X[:, i]
        print(f"{labels[i]:<20}{xi.min():6.0f}{xi.max():6.0f}{xi.max()-xi.min():7.0f}")

    full = np.mean([X[:, i].max() - X[:, i].min() for i in range(16)])
    print(f"\n平均レンジ = {full:.0f} / 1023")
    if full > 300:
        print("→ フルスイングOK. 駆動法は健全. 問題はNARMA10入力の出し方に限定.")
    else:
        print("→ 三角波でも張り付く. 本番スケッチの駆動法(往復通信)が原因.")

    print("sweep_states.csv に保存.")


if __name__ == "__main__":
    main()