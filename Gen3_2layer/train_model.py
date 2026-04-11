import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 警告を非表示にする（データが少ない時に出る警告対策）
warnings.filterwarnings('ignore')

# === 1. データの読み込み ===
file_path = "reservoir_dataset.csv"
try:
    df = pd.read_csv(file_path)
    print(f"データを読み込みました！ (総データ数: {len(df)}件)")
except FileNotFoundError:
    print(f"エラー: {file_path} が見つかりません。データ収集を先に行ってください。")
    exit()

# ラベル名と番号のマッピング
label_names = {0: 'Wood', 1: 'Plastic', 2: 'Metal', 3: 'None'}

# 収集したラベルの種類を確認
recorded_labels = df['label'].unique()
print("記録されているラベル:", [f"{lbl}:{label_names[lbl]}" for lbl in sorted(recorded_labels)])

# === 2. X（特徴量）と y（正解ラベル）の分割 ===
y = df['label'].values
X = df.drop('label', axis=1).values

# === 3. 学習用とテスト用に分割 ===
# データが少ない場合はテストに回す割合を調整 (デフォルトは 8:2)
# stratify=y は、各材質の割合が学習とテストで均等になるようにする魔法のパラメータ
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
except ValueError:
    # データが少なすぎたり、特定のラベルが1個しかない場合は層化抽出をオフにする
    print("※データが少ないため、ランダム分割で実行します。")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. 正規化 (StandardScaler) ===
# リザバー計算において極めて重要！微小な波形もAIが認識できるようにスケールを揃える
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 5. AIの学習 (Ridge分類器) ===
print("\nAIをトレーニング中...")
# alphaは正則化パラメータ。リザバーのノイズ過学習を防ぐ。
model = RidgeClassifier(alpha=1.0)
model.fit(X_train_scaled, y_train)

# === 6. テストデータで予測と精度評価 ===
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("="*30)
print(f"✨ 材質識別 精度: {accuracy * 100:.1f}% ✨")
print("="*30)

# === 7. 混同行列 (Confusion Matrix) の可視化 ===
# AIが「何を何と間違えたか」を分析する
cm = confusion_matrix(y_test, y_pred, labels=sorted(recorded_labels))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[label_names[lbl] for lbl in sorted(recorded_labels)],
            yticklabels=[label_names[lbl] for lbl in sorted(recorded_labels)])
plt.title('Physical Reservoir - Confusion Matrix\n(True vs Predicted)', fontsize=14)
plt.xlabel('Predicted Material (AI Answer)', fontsize=12)
plt.ylabel('True Material (Actual)', fontsize=12)
plt.tight_layout()
plt.show()