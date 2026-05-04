import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

PORT = 'COM4'  
BAUD = 115200
MAX_LEN = 200

data_E = deque(maxlen=MAX_LEN)
data_G = deque(maxlen=MAX_LEN)
data_H = deque(maxlen=MAX_LEN)
data_I = deque(maxlen=MAX_LEN) # ★追加

try:
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
except Exception as e:
    print(f"エラー: {PORT} が開けません。\n{e}")
    exit()

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
line_e, = ax.plot([], [], label='Node E (A0: Main)', color='blue', lw=2)
line_g, = ax.plot([], [], label='Node G (A1: Clipped)', color='cyan', lw=2)
line_h, = ax.plot([], [], label='Node H (A2: Cross-Talk)', color='green', lw=2)
line_i, = ax.plot([], [], label='Node I (A3: MOSFET)', color='red', lw=2) # ★追加

ax.set_xlim(0, MAX_LEN)
ax.set_ylim(0, 1024)
ax.legend(loc='upper right')
ax.set_title("Layer 1~4: Physical Reservoir Diagnostics")
ax.grid(True, linestyle='--', alpha=0.6)

def update(frame):
    while ser.in_waiting:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                vals = [int(v) for v in line.split(',')]
                if len(vals) >= 4:
                    data_E.append(vals[0])
                    data_G.append(vals[1])
                    data_H.append(vals[2])
                    data_I.append(vals[3]) # ★追加
        except:
            pass

    line_e.set_data(range(len(data_E)), data_E)
    line_g.set_data(range(len(data_G)), data_G)
    line_h.set_data(range(len(data_H)), data_H)
    line_i.set_data(range(len(data_I)), data_I) # ★追加
    return line_e, line_g, line_h, line_i

ani = animation.FuncAnimation(fig, update, interval=20, blit=True)
plt.tight_layout()
plt.show()
ser.close()