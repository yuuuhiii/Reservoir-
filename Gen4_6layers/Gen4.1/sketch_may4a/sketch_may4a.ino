void setup() {
  Serial.begin(115200);
  Serial.setTimeout(10); // シリアル通信の待機時間を短くする
  pinMode(9, OUTPUT);
}

void loop() {
  // PCからテスト問題（0〜255のデータ）が送られてくるのを待つ
  if (Serial.available() > 0) {
    String inputStr = Serial.readStringUntil('\n');
    int pwmValue = inputStr.toInt();

    // 回路へ入力（D9からPWM出力）
    analogWrite(9, constrain(pwmValue, 0, 255));

    // ★超重要：物理回路が反応し、ダムに水（電荷）が移動するのを待つ時間
    delay(10); 

    // リザバー（AIの脳）の状態を読み取る
    int nodeE = analogRead(A0);
    int nodeG = analogRead(A1);
    int nodeH = analogRead(A2);
    int nodeI = analogRead(A3);

    // PCへ計算結果を返信
    Serial.print(nodeE); Serial.print(",");
    Serial.print(nodeG); Serial.print(",");
    Serial.print(nodeH); Serial.print(",");
    Serial.println(nodeI);
  }
}