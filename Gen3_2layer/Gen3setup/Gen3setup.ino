// Deep Physical Reservoir Ver 3.0 - Data Acquisition (Leonardo対応版)
const int numChannels = 4;
const int pins[] = {A0, A1, A2, A3};

void setup() {
  // 通信速度を高速に設定
  Serial.begin(115200);

  // ※Leonardo特有の処理：シリアルポートが開くまで待機する
  while (!Serial) {
    ; // PC側でPythonやシリアルモニタが開くまでここで待機します
  }
}

void loop() {
  // 4つのノード(E, F, G, H)の電圧を読み取る
  for (int i = 0; i < numChannels; i++) {
    Serial.print(analogRead(pins[i]));
    
    // 最後のデータ以外はカンマで区切る
    if (i < numChannels - 1) {
      Serial.print(",");
    }
  }
  Serial.println(); // 改行して1セット完了
  
  // サンプリングレート調整（約200Hz = 5ms）
  delay(5); 
}