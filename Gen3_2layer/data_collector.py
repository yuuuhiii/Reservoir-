import serial
import time
import csv
import numpy as np

# === 設定 ===
SERIAL_PORT = 'COM4'  # Leonardoのポート番号を確認して書き換えてください
BAUD_RATE = 115200
RECORD_STEPS = 200    # 1回の計測ステップ数 (約1秒)
OUTPUT_FILE = "reservoir_dataset.csv"

def collect_data():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        print(f"{SERIAL_PORT} に接続しました。")
    except Exception as e:
        print(f"接続エラー: {e}")
        return

    # CSVのヘッダー作成（ファイルが存在しない場合のみ作成）
    try:
        with open(OUTPUT_FILE, 'x', newline='') as f:
            writer = csv.writer(f)
            header = ['label']
            for t in range(RECORD_STEPS):
                header += [f'E_{t}', f'F_{t}', f'G_{t}', f'H_{t}']
            writer.writerow(header)
            print(f"新規ファイル {OUTPUT_FILE} を作成しました。")
    except FileExistsError:
        print(f"既存のファイル {OUTPUT_FILE} にデータを追記します。")

    while True:
        print("\n" + "="*30)
        print("【計測モード】")
        print("0: 木材 (Wood)")
        print("1: プラスチック (Plastic)")
        print("2: 金属 (Metal)")
        print("3: 何もなし (None/Silence)")
        print("q: 終了して保存")
        print("="*30)
        
        label = input("記録するラベルの番号を入力してください: ")
        
        if label.lower() == 'q':
            break
        if label not in ['0', '1', '2', '3']:
            print(">> 無効な入力です。0, 1, 2, 3 のいずれかを入力してください。")
            continue

        label_names = { '0':'木材', '1':'プラスチック', '2':'金属', '3':'何もなし' }
        print(f"\n>> 現在のターゲット: 【{label_names[label]}】")
        
        if label == '3':
            print(">> ピエゾに触れず、静かにしていてください。")
        else:
            print(f">> 準備ができたらEnterを押し、すぐに {label_names[label]} を1秒間こすってください。")
            
        input("Press Enter to Start Recording...")

        buffer = []
        print("● 録画中...")
        
        # 指定ステップ分（200回）データを読み込む
        while len(buffer) < RECORD_STEPS:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    values = line.split(',')
                    if len(values) == 4:
                        # ADC値を数値として格納
                        buffer.append([int(v) for v in values])
                except (ValueError, UnicodeDecodeError):
                    continue

        # データの平坦化 (label, E0, F0, G0, H0, E1, F1, ...)
        flattened_row = [label]
        for step_data in buffer:
            flattened_row.extend(step_data)

        # CSVに1行追記
        with open(OUTPUT_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(flattened_row)
        
        # 現在のサンプル数をカウントして表示
        sample_count = sum(1 for _ in open(OUTPUT_FILE)) - 1
        print(f">> 保存完了！ 現在の合計データ数: {sample_count}件")

    ser.close()
    print("\nすべてのデータを保存しました。お疲れ様でした！")

if __name__ == "__main__":
    collect_data()