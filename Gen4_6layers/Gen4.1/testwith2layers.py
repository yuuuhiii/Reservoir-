import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# --- 設定 ---
PORT = 'COM4'  
BAUD = 115200
MAX_LEN = 200  # 画面に表示するデータ数（横軸の長さ）

# --- データ保存用のキュー ---
data_E = deque(maxlen=MAX_LEN)
data_F = deque(maxlen=MAX_LEN)
data_G = deque(maxlen=MAX_LEN)

# シリアル通信の開始
try:
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    print(f"{PORT} に接続しました。オシロスコープを起動します...")
except Exception as e:
    print(f"エラー: {PORT} が開けません。ArduinoIDEのシリアルモニタを閉じているか確認してください。\n{e}")
    exit()

# グラフの設定
fig, ax = plt.subplots(figsize=(10, 6))
line_e, = ax.plot([], [], label='Node E (A0: RC Filter)', color='blue', lw=2)
line_f, = ax.plot([], [], label='Node F (A1: Delayed)', color='cyan', lw=2)
line_g, = ax.plot([], [], label='Node G (A2: Clipped)', color='green', lw=2)

ax.set_xlim(0, MAX_LEN)
ax.set_ylim(0, 1024)
ax.legend(loc='upper right')
ax.set_title("Layer 1 & 2: Hardware Diagnosis")
ax.set_ylabel("ADC Value (0-1023)")
ax.grid(True, linestyle='--', alpha=0.6)

# アニメーション更新関数
def update(frame):
    while ser.in_waiting:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                vals = [int(v) for v in line.split(',')]
                if len(vals) >= 3:
                    data_E.append(vals[0])
                    data_F.append(vals[1])
                    data_G.append(vals[2])
        except Exception:
            pass # ノイズ等でパース失敗した場合は無視

    # データをグラフに反映
    line_e.set_data(range(len(data_E)), data_E)
    line_f.set_data(range(len(data_F)), data_F)
    line_g.set_data(range(len(data_G)), data_G)
    return line_e, line_f, line_g

# アニメーション開始
ani = animation.FuncAnimation(fig, update, interval=20, blit=True)
plt.tight_layout()
plt.show()

ser.close()