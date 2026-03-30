import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import sys

# --- 必須設定 ---
SERIAL_PORT = 'COM4'  # ※ご自身の環境に合わせて変更してください！
BAUD_RATE = 115200
WINDOW_SIZE = 200     # 画面に表示するデータの幅（時間）

# 8ノード分のデータキュー（初期値は0）
data_queues = [deque(np.zeros(WINDOW_SIZE), maxlen=WINDOW_SIZE) for _ in range(8)]

print(f"[{SERIAL_PORT}] に接続中...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    ser.dtr = False 
except Exception as e:
    print(f"シリアルポートのオープンに失敗しました: {e}")
    print("ポート番号（COM4など）が合っているか、Arduino IDEのシリアルモニタを閉じているか確認してください。")
    sys.exit()

# --- グラフ描画のセットアップ ---
fig, ax = plt.subplots(figsize=(12, 6))
fig.canvas.manager.set_window_title("Piezo Physical Reservoir Monitor")
ax.set_title("Piezo & Physical Reservoir Real-time Monitor", fontsize=16)
ax.set_xlim(0, WINDOW_SIZE)

# ★注目ポイント★
# ピエゾが発する電気は微小なため、最初はY軸を 0〜300 くらいに絞って拡大表示します。
# もし波が画面の上に突き抜けるほど元気なら、ここを 0〜1023 に変更してください。
ax.set_ylim(-10, 300) 

ax.set_xlabel("Time Step")
ax.set_ylabel("Voltage (0-1023)")
ax.grid(True)

lines = []
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
for i in range(8):
    line, = ax.plot([], [], label=f"N{i}", color=colors[i], alpha=0.8, linewidth=1.5)
    lines.append(line)
ax.legend(loc="upper left")

def update(frame):
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if not line: continue
            
            parts = line.split(',')
            # "N0:123" のような形式から数値だけを抽出
            values = [int(p.split(':')[1]) for p in parts if ':' in p]
            
            if len(values) == 9: # Orig + N0~N7
                current_state = values[1:9] # N0~N7の部分だけを抽出
                for i in range(8):
                    data_queues[i].append(current_state[i])
        except Exception:
            pass # 最初のノイズや変換エラーは無視
            
    # グラフの線を最新データで更新
    for i in range(8):
        lines[i].set_data(range(WINDOW_SIZE), data_queues[i])
    return lines

print("\n" + "="*50)
print("🚀 リアルタイムモニター起動！")
print("ピエゾ素子を指でこすったり、軽くコンコンと叩いたりしてみてください。")
print("="*50 + "\n")

ani = animation.FuncAnimation(fig, update, interval=20, blit=True, cache_frame_data=False)

try:
    plt.tight_layout()
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    ser.close()
    print("シリアルポートを閉じてモニターを終了しました。")