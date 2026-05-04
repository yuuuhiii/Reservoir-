float t = 0;

void setup() {
  Serial.begin(115200);
  pinMode(9, OUTPUT);
}

void loop() {
  // 緩やかなサイン波を作成してD9から出力 (PWM 0〜255)
  int outVal = (sin(t) + 1.0) * 127.5;
  analogWrite(9, outVal);

  // ブレッドボードからの信号を読み取る
  int nodeE = analogRead(A0); // Layer 1: なめらかな波
  int nodeF = analogRead(A1); // Layer 2-1: 遅れた波
  int nodeG = analogRead(A2); // Layer 2-2: 歪んだ波

  // Pythonへカンマ区切りで送信
  Serial.print(nodeE); Serial.print(",");
  Serial.print(nodeF); Serial.print(",");
  Serial.println(nodeG);

  t += 0.05; // 波の速さ
  delay(10); // 10ミリ秒ごとに計測
}