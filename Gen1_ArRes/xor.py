import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier # ★分類用のモデルに変更
from sklearn.metrics import accuracy_score

# --- 通信設定（今までと同じ） ---
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
        print("\nデータ収集を停止しました。XORタスクに移行します...")
    finally:
        ser.close()
except serial.SerialException as e:
    print(f"シリアル通信エラー: {e}")
    exit()

# --- ここから機械学習（遅延XORタスク） ---
reservoir_data = np.array(data_records)
if reservoir_data.shape[0] < 100:
    exit()

orig_signal = reservoir_data[:, 0]
X = reservoir_data[:, 1:9]

# 1. 元の信号を 0 と 1 のビット列に変換（127を閾値とする）
bits = np.where(orig_signal > 127, 1, 0)

# 2. 正解ラベルの作成：遅延XOR (現在 t のビット と 過去 t-1 のビット の排他的論理和)
# delay_step を大きくするほど、過去の記憶が必要になり難易度が跳ね上がります
delay_step = 1 
y_target = np.zeros_like(bits)
for i in range(delay_step, len(bits)):
    # Pythonの ^ 演算子でXORを計算
    y_target[i] = bits[i] ^ bits[i - delay_step] 

# 最初の方は過去データがないのでカット
X = X[delay_step:]
y_target = y_target[delay_step:]

# データを分割（前半70%で学習、後半30%でテスト）
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y_target[:split_idx], y_target[split_idx:]

# リッジ分類器で学習
model = RidgeClassifier(alpha=10.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 評価：正解率（Accuracy）を計算
accuracy = accuracy_score(y_test, y_pred)
print("="*40)
print(f"★ 遅延XORタスクの正解率: {accuracy * 100:.2f}%")
print("  (※ランダムに答えると50%です。100%に近いほど優秀！)")
print("="*40)

# --- 結果のグラフ化 ---
# 最初の100ステップだけを拡大して表示（見やすくするため）
plot_steps = min(100, len(y_test))

plt.figure(figsize=(12, 6))
# 元の入力ビット（参考）
plt.plot(bits[split_idx + delay_step : split_idx + delay_step + plot_steps] * 0.8 - 1.2, 
         label="Input Bits (0 or 1)", color='green', drawstyle='steps-pre', alpha=0.5)

# 正解のXOR出力と、AIの予測
plt.plot(y_test[:plot_steps], label="Target (XOR output)", linestyle='--', color='gray', drawstyle='steps-pre')
plt.plot(y_pred[:plot_steps], label=f"AI Prediction (Accuracy: {accuracy*100:.1f}%)", color='blue', alpha=0.7, drawstyle='steps-pre')

plt.title(f"Physical Reservoir Computing: Delayed XOR Task (Delay = {delay_step})")
plt.yticks([-1.2, -0.4, 0, 1], ['Input=0', 'Input=1', 'Output=0', 'Output=1'])
plt.xlabel("Time Step (Test Data Only)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()