import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# --- 通信設定 ---
SERIAL_PORT = 'COM4' 
BAUD_RATE = 115200
data_records = []

print(f"{SERIAL_PORT} に接続しています...")
try:
    ser = serial.Serial()
    ser.port = SERIAL_PORT
    ser.baudrate = BAUD_RATE
    ser.timeout = 2
    ser.dtr = False 
    ser.open()
    
    try:
        print("通信開始！データを取得しています... (15秒ほど待ってから Ctrl+C)")
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
        print("\nデータ収集を停止しました。未来予測フェーズに移行します...")
    finally:
        ser.close()
except serial.SerialException as e:
    print(f"シリアル通信エラー: {e}")
    exit()

# --- ここから機械学習（未来予測タスク） ---
reservoir_data = np.array(data_records)
if reservoir_data.shape[0] < 100:
    exit()

orig_signal = reservoir_data[:, 0]
X_all = reservoir_data[:, 1:9]

# ★未来予測のステップ数（どれくらい先を予測するか）
future_steps = 15 

# ★ここが魔法のコード（時間をズラす）
# 入力Xは「最初から、終点のfuture_steps手前まで」
X = X_all[:-future_steps] 
# 正解yは「future_steps先から、最後まで」の生のサイン波
y_target = orig_signal[future_steps:] 

# データを分割（前半70%で学習、後半30%でテスト）
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y_target[:split_idx], y_target[split_idx:]

# リッジ回帰で学習
model = Ridge(alpha=10.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse_score = mean_squared_error(y_test, y_pred)
print(f"★ 未来予測({future_steps}ステップ先)のMSEスコア: {mse_score:.4f}")

# --- 結果のグラフ化 ---
plt.figure(figsize=(10, 5))
plt.plot(y_test, label=f"Target (Actual Future Sine Wave)", linestyle='--', color='gray', alpha=0.7)
plt.plot(y_pred, label=f"AI Future Prediction (MSE: {mse_score:.4f})", color='blue')

plt.title(f"Physical Reservoir Computing: Predicting {future_steps} Steps Ahead")
plt.xlabel("Time Step (Test Data Only)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()