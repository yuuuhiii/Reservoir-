float t = 0;

void setup() {
  Serial.begin(115200);
  pinMode(9, OUTPUT);
}

void loop() {
  int outVal = (sin(t) + 1.0) * 127.5;
  analogWrite(9, outVal);

  int nodeE = analogRead(A0); // 青線
  int nodeG = analogRead(A1); // 水色線
  int nodeH = analogRead(A2); // 緑線
  int nodeI = analogRead(A3); // ★追加: 赤線 (MOSFET出力)

  Serial.print(nodeE); Serial.print(",");
  Serial.print(nodeG); Serial.print(",");
  Serial.print(nodeH); Serial.print(",");
  Serial.println(nodeI);

  t += 0.05;
  delay(10);
}