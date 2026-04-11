import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np

# === ユーザー設定 ===
SERIAL_PORT = 'COM4'  # Leonardoのポートに合わせてください
BAUD_RATE = 115200
MAX_POINTS = 300

data_e = deque([0] * MAX_POINTS, maxlen=MAX_POINTS) 
data_f = deque([0] * MAX_POINTS, maxlen=MAX_POINTS) 
data_g = deque([0] * MAX_POINTS, maxlen=MAX_POINTS) 
data_h = deque([0] * MAX_POINTS, maxlen=MAX_POINTS) 

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(1)
    print("4連装オシロスコープを起動します...")
except Exception as e:
    print(f"シリアル通信エラー: {e}")
    exit()

# グラフを縦に4つ並べる設定
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Deep Physical Reservoir - 4 Channel Micro-Scope", fontsize=14)

# 各グラフの初期設定（Y軸は自動調整にするため固定しない）
lines = []
colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
labels = ["Node E (Short)", "Node F (Mid)", "Node G (Long)", "Node H (Chaos)"]
data_refs = [data_e, data_f, data_g, data_h]

for i in range(4):
    line, = axs[i].plot(data_refs[i], color=colors[i], label=labels[i])
    axs[i].legend(loc="upper right")
    axs[i].grid(True, linestyle=':', alpha=0.6)
    lines.append(line)

axs[3].set_xlabel("Time (steps)")

def update(frame):
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            values = line.split(',')
            if len(values) == 4:
                data_e.append(int(values[0]))
                data_f.append(int(values[1]))
                data_g.append(int(values[2]))
                data_h.append(int(values[3]))
        except ValueError:
            pass

    # データを更新し、各グラフのY軸スケールを自動フィットさせる
    for i in range(4):
        lines[i].set_ydata(data_refs[i])
        axs[i].relim()           # データに基づいてリミットを再計算
        axs[i].autoscale_view()  # ビューを自動調整

    # === 定量評価：100フレーム（約1秒）ごとに相関行列を計算して表示 ===
    if frame % 100 == 0 and len(data_e) == MAX_POINTS:
        # 4つのリストをNumPy配列に変換
        matrix = np.array([data_e, data_f, data_g, data_h], dtype=float)
        
        # 分散が0（完全に平坦）だと計算エラーになるため、微小なノイズを足して回避
        matrix += np.random.normal(0, 1e-5, matrix.shape) 
        
        # 相関行列を計算
        corr_matrix = np.corrcoef(matrix)
        
        print(f"--- 信号分離度 (Step: {frame}) ---")
        print("      [E]    [F]    [G]    [H]")
        labels = ['[E]', '[F]', '[G]', '[H]']
        for i in range(4):
            row_str = f"{labels[i]} "
            for j in range(4):
                # 小数点以下2桁でフォーマット
                row_str += f"{corr_matrix[i, j]:6.2f} "
            print(row_str)
        print("-" * 32)

    return lines

ani = animation.FuncAnimation(fig, update, interval=10, blit=False) # 自動スケールのためblit=False
plt.tight_layout()
plt.show()

ser.close()