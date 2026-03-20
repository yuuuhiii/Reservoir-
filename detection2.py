import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
import time

# 1. データの自動ダウンロードと準備
print("Downloading FordA dataset...")
train_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TRAIN.tsv"
test_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TEST.tsv"

train_data = pd.read_csv(train_url, sep='\t', header=None).values
test_data = pd.read_csv(test_url, sep='\t', header=None).values

y_train, X_train_raw = train_data[:, 0], train_data[:, 1:]
y_test, X_test_raw = test_data[:, 0], test_data[:, 1:]

# 2. デバイスの設定 (ここで 'cuda' と出れば成功！)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 3. リザーバーの構築 (GPUのパワーを見るため、ノード数を2000に設定)
n_res = 2000
print(f"Building Reservoir with {n_res} nodes...")
np.random.seed(42)
W_in_np = np.random.rand(n_res, 1) * 2 - 1
W_res_np = np.random.rand(n_res, n_res) * 2 - 1

radius = np.max(np.abs(np.linalg.eigvals(W_res_np)))
W_res_np = W_res_np * (0.95 / radius)

# NumPyの配列を、GPUで計算できるPyTorchのテンソルに変換
W_in = torch.tensor(W_in_np, dtype=torch.float32).to(device)
W_res = torch.tensor(W_res_np, dtype=torch.float32).to(device)

# 4. 特徴量抽出関数 (ピークホールド / Max Pooling版)
def extract_reservoir_features_gpu(X_raw):
    X_pt = torch.tensor(X_raw, dtype=torch.float32).to(device)
    n_samples, n_steps = X_pt.shape
    
    x_state = torch.zeros((n_samples, n_res), dtype=torch.float32, device=device)
    
    # 振動の「最大値（ピーク）」を記録するためのテンソル
    # tanh関数の最小値は-1なので、初期値を-1.0にしておく
    max_states = torch.full((n_samples, n_res), -1.0, dtype=torch.float32, device=device)
    
    alpha = 0.3
    input_scale = 0.01
    
    for t in range(n_steps):
        u_t = X_pt[:, t].reshape(-1, 1) * input_scale
        
        # 物理モデル: (1 - α)*過去の状態 + α*新しい応答
        update = torch.tanh(torch.matmul(u_t, W_in.T) + torch.matmul(x_state, W_res.T))
        x_state = (1 - alpha) * x_state + alpha * update
        
        # 平均するのではなく、各ノードの「最も波が高かった瞬間の値」を上書き保存する
        max_states = torch.maximum(max_states, x_state)
        
    return max_states.cpu().numpy()

# 5. 実行と時間計測
print("Extracting features from training data (GPU)...")
start_time = time.time()
X_train_features = extract_reservoir_features_gpu(X_train_raw)
print(f" -> Train Extraction Time: {time.time() - start_time:.2f} seconds")

print("Extracting features from testing data (GPU)...")
start_time = time.time()
X_test_features = extract_reservoir_features_gpu(X_test_raw)
print(f" -> Test Extraction Time: {time.time() - start_time:.2f} seconds")

# 6. 出力層の学習と評価 (ここはCPUで一瞬で終わります)
print("Training the readout layer...")
classifier = RidgeClassifier(alpha=1e-3)
classifier.fit(X_train_features, y_train)

predictions = classifier.predict(X_test_features)
accuracy = accuracy_score(y_test, predictions)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")