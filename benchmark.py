import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# --- 通信設定 ---
SERIAL_PORT = 'COM4'  # 環境に合わせて変更してください
BAUD_RATE = 115200
data_records = []

print(f"[{SERIAL_PORT}] に接続中...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    ser.dtr = False 
    
    print("通信開始！ランダムノイズを入力して物理ダイナミクスを励起中...")
    print("★ 20〜30秒ほど待ってから [Ctrl+C] を押して学習をスタートしてください ★")
    try:
        while True:
            line = ser.readline().decode('utf-8').strip()
            if not line: continue
            try:
                parts = line.split(',')
                values = [int(p.split(':')[1]) for p in parts if ':' in p]
                if len(values) == 9: 
                    data_records.append(values)
            except IndexError:
                pass
    except KeyboardInterrupt:
        print("\nデータ収集完了！NARMA10の学習と評価を開始します...")
    finally:
        ser.close()
except Exception as e:
    print(f"エラー: {e}")
    exit()

# --- 機械学習（NARMA10タスク） ---
reservoir_data = np.array(data_records)
if reservoir_data.shape[0] < 200:
    print("データ不足です。最低200ステップ必要です。")
    exit()

# 1. 入力データの正規化（0.0〜0.5）
raw_input = reservoir_data[:, 0]
u = (raw_input - np.min(raw_input)) / (np.max(raw_input) - np.min(raw_input)) * 0.5
X_all = reservoir_data[:, 1:9] # N0〜N7の8ノード

# 2. NARMA10の正解ラベル生成（爆発防止リミッター付き）
def generate_narma10(u_signal):
    length = len(u_signal)
    y = np.zeros(length)
    for t in range(10): y[t] = 0.1 
    for t in range(9, length - 1):
        y_sum = np.sum(y[t-9 : t+1]) 
        next_y = 0.3 * y[t] + 0.05 * y[t] * y_sum + 1.5 * u_signal[t-9] * u_signal[t] + 0.1
        y[t+1] = np.clip(next_y, 0.0, 1.0) # 強制リミッター
    return y

y_target = generate_narma10(u)

# 過去10ステップ分の助走データを捨てる
X = X_all[10:]
y_target = y_target[10:]

# 訓練データ(70%)とテストデータ(30%)に分割
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y_target[:split_idx], y_target[split_idx:]

# 学習と予測
model = Ridge(alpha=10.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# スコア計算
mse = mean_squared_error(y_test, y_pred)
var_y = np.var(y_test)
nrmse = np.sqrt(mse / var_y)

print("="*50)
print(f"★ Grid Reservoir Ver 2.0 NARMA10 NRMSE: {nrmse:.4f}")
print("="*50)

# --- 結果のグラフ化（2段構え） ---
plot_steps = min(150, len(y_test))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 上段：AIの予測精度
ax1.plot(y_test[:plot_steps], label="Target (NARMA10)", linestyle='--', color='black', linewidth=2)
ax1.plot(y_pred[:plot_steps], label=f"AI Prediction (NRMSE: {nrmse:.4f})", color='red', alpha=0.8, linewidth=2)
ax1.set_title("Top: AI Performance on NARMA10 Task", fontsize=14)
ax1.set_ylabel("Value")
ax1.legend()
ax1.grid(True)

# 下段：リザバー内部の波形（N0〜N7）
for i in range(8):
    ax2.plot(X_test[:plot_steps, i], label=f"N{i}", alpha=0.7)
ax2.set_title("Bottom: Internal Dynamics of Grid Reservoir (Chaos & Memory)", fontsize=14)
ax2.set_xlabel("Time Step (Test Data)")
ax2.set_ylabel("Voltage (0-1023)")
ax2.legend(loc='upper right', ncol=4)
ax2.grid(True)

plt.tight_layout()
plt.show()