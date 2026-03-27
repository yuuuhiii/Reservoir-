import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

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
        # NARMA10はデータ数が命です。長めに取ります。
        print("通信開始！データを取得しています... (20〜30秒ほど待ってから Ctrl+C)")
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
        print("\nデータ収集を停止しました。NARMA10タスクに移行します...")
    finally:
        ser.close()
except serial.SerialException as e:
    print(f"シリアル通信エラー: {e}")
    exit()

# --- ここから機械学習（NARMA10タスク） ---
reservoir_data = np.array(data_records)
if reservoir_data.shape[0] < 200:
    print("データが少なすぎます。NARMA10には最低200ステップ以上のデータが必要です。")
    exit()

# 1. 入力データの正規化（0.0〜0.5の範囲にスケーリング）
# Arduinoからはランダムな値（例: 0〜255）が送られてくる前提
raw_input = reservoir_data[:, 0]
u = (raw_input - np.min(raw_input)) / (np.max(raw_input) - np.min(raw_input)) * 0.5
X_all = reservoir_data[:, 1:9]

# 2. NARMA10の正解ラベル(y)を数式通りに生成する関数
def generate_narma10(u_signal):
    length = len(u_signal)
    y = np.zeros(length)
    # 最初の10ステップは計算できないので0.1で初期化
    for t in range(10):
        y[t] = 0.1 
    
    # 10ステップ目以降からNARMA10の数式を適用
    for t in range(9, length - 1):
        # 過去10ステップのyの合計
        y_sum = np.sum(y[t-9 : t+1]) 
        # 例のヤバい数式
        y[t+1] = 0.3 * y[t] + 0.05 * y[t] * y_sum + 1.5 * u_signal[t-9] * u_signal[t] + 0.1
    return y

y_target = generate_narma10(u)

# 過去10ステップ分の「助走」データは捨てる
X = X_all[10:]
y_target = y_target[10:]

# データを分割（前半70%で学習、後半30%でテスト）
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y_target[:split_idx], y_target[split_idx:]

# リッジ回帰で学習
model = Ridge(alpha=10.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 評価：NRMSE (Normalized Root Mean Square Error) を計算
# NARMAタスクではMSEをデータの分散で割ったNRMSEが標準的な評価指標です
mse = mean_squared_error(y_test, y_pred)
var_y = np.var(y_test)
nrmse = np.sqrt(mse / var_y)

print("="*40)
print(f"★ NARMA10タスクの NRMSEスコア: {nrmse:.4f}")
print("  (※ 0.0に近いほど優秀。0.3を下回れば論文レベルの超優秀ハードウェア！)")
print("="*40)

# --- 結果のグラフ化 ---
# 最初の150ステップだけを拡大して表示
plot_steps = min(150, len(y_test))

plt.figure(figsize=(12, 6))
plt.plot(y_test[:plot_steps], label="Target (NARMA10 Output)", linestyle='--', color='gray', alpha=0.8)
plt.plot(y_pred[:plot_steps], label=f"AI Prediction (NRMSE: {nrmse:.4f})", color='purple', alpha=0.8)

plt.title("Physical Reservoir Computing: NARMA10 Task (The Final Boss)")
plt.xlabel("Time Step (Test Data Only)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()