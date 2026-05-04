import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# --- 設定 ---
PORT = 'COM4'  
BAUD = 115200
DATA_LENGTH = 1000  # テストするデータの長さ（1000問）
TRAIN_LEN = 800     # そのうち学習に使う量（800問）
TEST_LEN = 200      # 採点に使う量（200問）

print("🤖 NARMA10データセットを生成中...")
# NARMA10の生成（入力uは0〜0.5のランダム値）
np.random.seed(42)
u = np.random.uniform(0, 0.5, DATA_LENGTH)
y_target = np.zeros(DATA_LENGTH)

for t in range(9, DATA_LENGTH - 1):
    # NARMA10の複雑な計算式（これが正解データ）
    y_target[t+1] = 0.3 * y_target[t] + 0.05 * y_target[t] * np.sum(y_target[t-9:t+1]) + 1.5 * u[t-9] * u[t] + 0.1

# Arduinoに送るために、入力u(0〜0.5)をPWM値(0〜255)に変換
pwm_inputs = (u * 2 * 255).astype(int)

# --- ハードウェア通信開始 ---
print(f"🔌 Arduino ({PORT}) と通信を開始します...")
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2) # Arduinoの再起動待ち
except Exception as e:
    print(f"❌ エラー: Arduinoに接続できません。ポート番号を確認してください。\n{e}")
    exit()

states = [] # 物理回路の反応（特徴量）を保存するリスト

print("🚀 物理リザバーへデータを流し込み中... (約10〜20秒かかります)")
ser.reset_input_buffer()

for i in range(DATA_LENGTH):
    # PCからArduinoへ問題を送信
    ser.write(f"{pwm_inputs[i]}\n".encode('utf-8'))
    
    # Arduino（物理回路）からの反応を受信
    line = ser.readline().decode('utf-8').strip()
    
    try:
        vals = [int(v) for v in line.split(',')]
        if len(vals) == 4:
            states.append(vals)
        else:
            states.append([0, 0, 0, 0]) # エラー時のダミー
    except:
        states.append([0, 0, 0, 0])
        
    # 進捗表示
    if (i + 1) % 100 == 0:
        print(f"  ... {i + 1} / {DATA_LENGTH} ステップ完了")

ser.close()
print("✅ データ収集完了！")

# --- AIの学習と評価（Readout学習） ---
print("🧠 scikit-learnで学習と採点を行っています...")
states = np.array(states)

# 学習用データとテスト用データに分割
X_train = states[10:TRAIN_LEN]  # 最初の10歩は安定しないので捨てる
y_train = y_target[10:TRAIN_LEN]
X_test = states[TRAIN_LEN:]
y_test = y_target[TRAIN_LEN:]

# Ridge回帰（線形モデル）で学習
model = Ridge(alpha=10.0)
model.fit(X_train, y_train)

# テストデータで予測
y_pred = model.predict(X_test)

# スコア（NRMSE）の計算：0に近いほど優秀！
mse = mean_squared_error(y_test, y_pred)
variance = np.var(y_test)
nrmse = np.sqrt(mse / variance)

print("\n" + "="*40)
print(f"🏆 ベンチマーク結果 (NRMSE スコア): {nrmse:.4f}")
print("   (0.0に近いほど完璧。0.4以下なら物理リザバーとして超優秀です！)")
print("="*40 + "\n")

# --- 結果のグラフ化 ---
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Target (True Answer)", color="black", linestyle="--", linewidth=2)
plt.plot(y_pred, label="Prediction (Reservoir Output)", color="red", linewidth=2)
plt.title(f"NARMA10 Benchmark Task (NRMSE: {nrmse:.4f})")
plt.xlabel("Time Step (Test Data)")
plt.ylabel("NARMA Value")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.tight_layout()
plt.show()