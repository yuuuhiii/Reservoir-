import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# --- 設定 ---
SERIAL_PORT = 'COM4'  # 環境に合わせて変更してください
BAUD_RATE = 115200
WINDOW_SIZE = 150     # 画面に表示するデータ点数（横軸の長さ）
NUM_NODES = 8         # N0〜N7の8ノード

# --- データ格納用のキュー（右から入れて左から押し出す） ---
# 8つのノードそれぞれのデータを保存するリストのリスト
data_queues = [deque(np.zeros(WINDOW_SIZE), maxlen=WINDOW_SIZE) for _ in range(NUM_NODES)]

# --- AIのダミー推論関数（後でここを書き換えます） ---
def predict_material(current_state):
    """
    本当はここで model.predict([current_state]) を行います。
    今はダミーとして、N1の電圧が高いか低いかで適当な判定を返します。
    """
    n1_voltage = current_state[1]
    if n1_voltage > 512:
        return "WOOD (木材)", "green"
    else:
        return "PLASTIC (プラ)", "blue"

# --- シリアル通信のセットアップ ---
print(f"{SERIAL_PORT} に接続しています...")
try:
    ser = serial.Serial()
    ser.port = SERIAL_PORT
    ser.baudrate = BAUD_RATE
    ser.timeout = 0.1 # リアルタイム描画を止めないように短いタイムアウトを設定
    ser.dtr = False 
    ser.open()
    time.sleep(2)
except serial.SerialException as e:
    print(f"シリアル通信エラー: {e}")
    exit()

# --- グラフのセットアップ ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Physical Reservoir Real-time Monitor", fontsize=16)
ax.set_xlim(0, WINDOW_SIZE)
ax.set_ylim(-100, 1100) # ArduinoのADCは0〜1023
ax.set_xlabel("Time Step")
ax.set_ylabel("Voltage (0-1023)")
ax.grid(True, linestyle='--', alpha=0.6)

# 8本の波形ラインを描画用に準備
lines = []
colors = ['red', 'orange', 'gold', 'green', 'blue', 'cyan', 'purple', 'magenta']
for i in range(NUM_NODES):
    line, = ax.plot([], [], label=f"Node {i}", color=colors[i], alpha=0.8)
    lines.append(line)

ax.legend(loc="upper left")

# AIの判定結果をデカデカと表示するためのテキストボックス
prediction_text = ax.text(0.5, 0.95, "Waiting for data...", 
                          transform=ax.transAxes, fontsize=24, fontweight='bold', 
                          ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))

# --- アニメーション更新関数（この関数が高速でループ実行されます） ---
def update(frame):
    current_state = None
    
    # Arduinoから届いているデータがあるだけ読み込む（溜まっている分を消化）
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue
            
            parts = line.split(',')
            values = [int(p.split(':')[1]) for p in parts if ':' in p]
            
            if len(values) == NUM_NODES + 1: # Orig(1個) + N0~N7(8個) = 9個
                current_state = values[1:9] # N0〜N7を現在の状態として取得
                
                # 各ノードのキューに最新データを追加
                for i in range(NUM_NODES):
                    data_queues[i].append(current_state[i])
        except (IndexError, ValueError, UnicodeDecodeError):
            pass # 通信エラーや文字化けは無視
            
    # 新しいデータが入っていれば、AIの判定を行いグラフを更新
    if current_state is not None:
        # AIに現在のリザバー状態を渡して判定！
        material, text_color = predict_material(current_state)
        prediction_text.set_text(f"AI Detection: {material}")
        prediction_text.set_color(text_color)
        
        # グラフの線を最新データに更新
        for i in range(NUM_NODES):
            lines[i].set_data(range(WINDOW_SIZE), data_queues[i])
            
    return lines + [prediction_text]

# --- アニメーションの開始 ---
print("リアルタイム推論モニターを起動しました！ (終了するにはウィンドウを閉じてください)")
ani = animation.FuncAnimation(fig, update, interval=20, blit=True, cache_frame_data=False)

try:
    plt.show() # グラフウィンドウを開いてループ開始
finally:
    ser.close()
    print("通信を終了しました。")