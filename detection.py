import numpy as np
import pandas as pd
import urllib.request
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. FordAデータセットの自動ダウンロードと読み込み
print("Downloading FordA dataset...")
train_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TRAIN.tsv"
test_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TEST.tsv"

train_data = pd.read_csv(train_url, sep='\t', header=None).values
test_data = pd.read_csv(test_url, sep='\t', header=None).values

# データの分割 (1列目がラベル、2列目以降が500ステップの時系列データ)
# ラベルは 1 (正常) と -1 (異常)
y_train, X_train_raw = train_data[:, 0], train_data[:, 1:]
y_test, X_test_raw = test_data[:, 0], test_data[:, 1:]

print(f"Train data shape: {X_train_raw.shape}, Test data shape: {X_test_raw.shape}")

# 2. リザーバーの構築
n_res = 300 # 特徴量をより豊かにするためノード数を300に設定
np.random.seed(42)
W_in = np.random.rand(n_res, 1) * 2 - 1
W_res = np.random.rand(n_res, n_res) * 2 - 1

# スペクトル半径の調整 (過去の記憶の残り具合を調整)
radius = np.max(np.abs(np.linalg.eigvals(W_res)))
W_res = W_res * (0.95 / radius)

# 3. リザーバーによる特徴量抽出関数
def extract_reservoir_features(X_raw):
    """各時系列データをリザーバーに通し、最後の状態を特徴量として取得する"""
    n_samples, n_steps = X_raw.shape
    features = np.zeros((n_samples, n_res))
    
    for i in range(n_samples):
        x = np.zeros(n_res)
        # 500ステップの波形を順番にリザーバーに入力
        for t in range(n_steps):
            x = np.tanh(np.dot(W_in, [X_raw[i, t]]) + np.dot(W_res, x))
        # 波形を全て読み終わった「最後のリザーバーの状態」を、この波形の特徴量とする
        features[i, :] = x
        
    return features

print("Extracting features from training data...")
X_train_features = extract_reservoir_features(X_train_raw)

print("Extracting features from testing data...")
X_test_features = extract_reservoir_features(X_test_raw)

# 4. 出力層の学習と評価 (分類タスクなので RidgeClassifier を使用)
print("Training the readout layer...")
classifier = RidgeClassifier(alpha=1e-3)
classifier.fit(X_train_features, y_train)

predictions = classifier.predict(X_test_features)
accuracy = accuracy_score(y_test, predictions)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")

# 5. 実際の波形の可視化 (正常と異常を1つずつ比較)
plt.figure(figsize=(10, 5))

# 正常な波形をプロット (ラベルが1のもの)
normal_idx = np.where(y_train == 1)[0][0]
plt.plot(X_train_raw[normal_idx], label="Normal Engine (Class 1)", color="blue", alpha=0.8)

# 異常な波形をプロット (ラベルが-1のもの)
abnormal_idx = np.where(y_train == -1)[0][0]
plt.plot(X_train_raw[abnormal_idx], label="Abnormal Engine (Class -1)", color="red", alpha=0.8, linestyle="--")

plt.title("FordA Dataset: Normal vs Abnormal Engine Noise")
plt.xlabel("Time Steps")
plt.ylabel("Sensor Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()