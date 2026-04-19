import serial
import time
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === 設定 ===
SERIAL_PORT = 'COM4'  # 環境に合わせて変更
BAUD_RATE = 115200
HISTORY_LEN = 100

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)
    print("🔌 6次元オシロスコープ (Node I ズーム搭載版) 接続完了！")
except Exception as e:
    print(f"接続エラー: {e}")
    exit()

# データ格納庫
data = {k: [0]*HISTORY_LEN for k in ['Input', 'E', 'F', 'G', 'H', 'I', 'J']}

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
fig.canvas.manager.set_window_title('Physical Reservoir - 6D Scope')

# [上段]
line_in, = ax1.plot(data['Input'], label='Input (Target)', color='gray', linestyle='--')
line_e, = ax1.plot(data['E'], label='Node E (Fast Amp)', color='blue', linewidth=2)
line_f, = ax1.plot(data['F'], label='Node F (RC Delay)', color='cyan')
ax1.set_ylim(0, 1024); ax1.legend(loc='upper right'); ax1.set_title('Layer 1 & 2: Fast Dynamics')

# [中段]
line_g, = ax2.plot(data['G'], label='Node G (Threshold)', color='green')
line_h, = ax2.plot(data['H'], label='Node H (Cross-Talk)', color='purple')
ax2.set_ylim(0, 1024); ax2.legend(loc='upper right'); ax2.set_title('Layer 2: Nonlinear Dynamics')

# [下段] 魔改造：Node I専用の「右側Y軸（ズーム）」を追加
ax3_i = ax3.twinx()  # 右側に独立した目盛りを作る
line_j, = ax3.plot(data['J'], label='Node J (Chaos Feedback)', color='red', linewidth=2)
line_i, = ax3_i.plot(data['I'], label='Node I (Deep Memory 1000μF)', color='darkgreen', linewidth=3)

ax3.set_ylim(0, 1024)
ax3.set_ylabel('Node J (0-1024)', color='red')
ax3_i.set_ylabel('Node I (Auto Zoom)', color='darkgreen')

# 凡例の結合
lines = [line_i, line_j]
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc='upper right')
ax3.set_title('Layer 3: Deep Memory (Auto-Scaled) & Recurrent Chaos')

def update(frame):
    t = time.time()
    base_val = 125 + 75 * math.sin(2 * math.pi * t / 3.0) 
    target = int(base_val + random.randint(-5, 5)) 
    
    ser.write(f"{target}\n".encode('utf-8'))
    
    start_time = time.time()
    response = ""
    while time.time() - start_time < 0.2:
        if ser.in_waiting > 0:
            response = ser.readline().decode('utf-8').strip()
            break
            
    if response:
        try:
            vals = list(map(int, response.split(',')))
            if len(vals) == 6:
                data['Input'].pop(0); data['Input'].append(target * 4)
                data['E'].pop(0); data['E'].append(vals[0])
                data['F'].pop(0); data['F'].append(vals[1])
                data['G'].pop(0); data['G'].append(vals[2])
                data['H'].pop(0); data['H'].append(vals[3])
                data['I'].pop(0); data['I'].append(vals[4])
                data['J'].pop(0); data['J'].append(vals[5])
                
                line_in.set_ydata(data['Input'])
                line_e.set_ydata(data['E'])
                line_f.set_ydata(data['F'])
                line_g.set_ydata(data['G'])
                line_h.set_ydata(data['H'])
                line_i.set_ydata(data['I'])
                line_j.set_ydata(data['J'])
                
                # Node Iの微小な変化を見逃さないための動的ズーム計算
                min_i, max_i = min(data['I']), max(data['I'])
                if max_i == min_i: # 変化が全くない時のエラー回避
                    max_i += 1; min_i -= 1
                ax3_i.set_ylim(min_i - 5, max_i + 5) # 波の上下に余白を持たせて追従
                
        except ValueError:
            pass
            
    return line_in, line_e, line_f, line_g, line_h, line_i, line_j

ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()

ser.write("0\n".encode('utf-8'))
ser.close()
print("実験終了")