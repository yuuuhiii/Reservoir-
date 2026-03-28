void setup() {
  Serial.begin(115200);
  pinMode(9, OUTPUT);
}

void loop() {
  // 0〜255の完全なランダム値（ノイズ）を生成して出力
  int pwmValue = random(0, 256); 
  analogWrite(9, pwmValue);
  delay(2);

  int node0 = analogRead(A0);
  int node1 = analogRead(A1);
  int node2 = analogRead(A2);
  int node3 = analogRead(A3);
  int node4 = analogRead(A4);
  int node5 = analogRead(A5);
  int node6 = analogRead(A6);
  int node7 = analogRead(A7);

  int originalSignal = map(pwmValue, 0, 255, 0, 1023);

  Serial.print("Orig:"); Serial.print(originalSignal); Serial.print(",");
  Serial.print("N0:"); Serial.print(node0); Serial.print(",");
  Serial.print("N1:"); Serial.print(node1); Serial.print(",");
  Serial.print("N2:"); Serial.print(node2); Serial.print(",");
  Serial.print("N3:"); Serial.print(node3); Serial.print(",");
  Serial.print("N4:"); Serial.print(node4); Serial.print(",");
  Serial.print("N5:"); Serial.print(node5); Serial.print(",");
  Serial.print("N6:"); Serial.print(node6); Serial.print(",");
  Serial.print("N7:"); Serial.println(node7);

  delay(10);
}