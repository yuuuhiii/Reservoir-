# auto_tester.py
import serial
import time
import random

# ポート番号は環境に合わせて変更してください
SERIAL_PORT = 'COM4' 
BAUD_RATE = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2) # Arduinoの再起動待ち
    print("🤖 HILシステム接続完了！リザバーへの自動注入を開始します...")
except Exception as e:
    print(f"接続エラー: {e}")
    exit()

try:
    while True:
        # 0〜255のランダムな波形(電圧)を生成
        target_voltage = random.randint(0, 255)
        
        # Arduinoへ送信 ("128\n" のような形式)
        ser.write(f"{target_voltage}\n".encode('utf-8'))
        
        # Arduinoからの返答(リザバーの計算結果)を待つ
        start_time = time.time()
        while time.time() - start_time < 0.5:
            if ser.in_waiting > 0:
                response = ser.readline().decode('utf-8').strip()
                # ターミナルに結果を美しく表示
                print(f"Input: {target_voltage:3d} ➔ Reservoir State [E, F, G, H, I, J]: {response}")
                break
                
        time.sleep(0.05) # 送信間隔

except KeyboardInterrupt:
    analogWrite_zero = "0\n"
    ser.write(analogWrite_zero.encode('utf-8'))
    ser.close()
    print("\nシステムを安全に停止しました。")