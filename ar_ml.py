import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error # ★評価用の関数を追加

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
        print("通信開始！データを取得しています... (15秒ほど待ってから Ctrl+C を押してください)")
        while True:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue
            try:
                parts = line.split(',')
                values = [int(p.split(':')[1]) for p in parts if ':' in p]
                # ★Orig1個 + ノード8個 = 合計9個のデータが揃っているか確認
                if len(values) == 9: 
                    data_records.append(values)
            except IndexError:
                pass
    except KeyboardInterrupt:
        print("\nデータ収集を停止しました。機械学習フェーズに移行します...")
    finally:
        ser.close()

except serial.SerialException as e:
    print(f"シリアル通信エラー: {e}")
    exit()

# --- ここから機械学習（リードアウト層の学習） ---
reservoir_data = np.array(data_records)

if reservoir_data.shape[0] < 100:
    print("データが少なすぎます。もう少し長く収集してください。")
    exit()

orig_signal = reservoir_data[:, 0]
# ★入力データをN0〜N7までの8列分に変更
X = reservoir_data[:, 1:9] 

threshold = (np.max(orig_signal) + np.min(orig_signal)) / 2
y_target = np.where(orig_signal > threshold, 1, -1)

split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y_target[:split_idx], y_target[split_idx:]

model = Ridge(alpha=10.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ★スコア（MSE）の計算と表示
# 正解の波(y_test)と、AIが作った波(y_pred)のズレの大きさを計算します。
# 0に近いほど、完璧な四角（誤差ゼロ）という意味になります！
mse_score = mean_squared_error(y_test, y_pred)
print("="*40)
print(f"★ 今回の物理リザバーのMSEスコア: {mse_score:.4f}")
print("  (※スコアは 0.0 に近いほど優秀です！)")
print("="*40)

# --- 結果のグラフ化 ---
plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Target (Ideal Square Wave)", linestyle='--', color='gray', alpha=0.7)
plt.plot(y_pred, label=f"AI Prediction (MSE: {mse_score:.4f})", color='red')

plt.title("Physical Reservoir Computing: Sine to Square Wave Task")
plt.xlabel("Time Step (Test Data Only)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()