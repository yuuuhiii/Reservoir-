/*
 * NARMA10 連続駆動スケッチ (Arduino Leonardo)
 * 入力をバッチ受信し, 定常を待たず2ms間隔で流す = フェーディングメモリ
 */

const int PIN_PWM = 9;
const int NODES[4] = {A0, A1, A2, A3};

const int STEP_INTERVAL_MS = 6;   // 短いほど記憶が残る
const int MAX_BATCH = 64;
int pwmBuf[MAX_BATCH];

void setup() {
  Serial.begin(115200);
  pinMode(PIN_PWM, OUTPUT);
  while (!Serial) { ; }
  analogWrite(PIN_PWM, 0);
}

void loop() {
  if (Serial.available() > 0) {
    String s = Serial.readStringUntil('\n');
    s.trim();
    if (s.length() == 0) return;

    int n = 0, start = 0;
    while (n < MAX_BATCH) {
      int comma = s.indexOf(',', start);
      String tok = (comma < 0) ? s.substring(start) : s.substring(start, comma);
      pwmBuf[n++] = constrain(tok.toInt(), 0, 255);
      if (comma < 0) break;
      start = comma + 1;
    }

    for (int i = 0; i < n; i++) {
      analogWrite(PIN_PWM, pwmBuf[i]);
      delay(STEP_INTERVAL_MS);

      int v0 = analogRead(NODES[0]);
      int v1 = analogRead(NODES[1]);
      int v2 = analogRead(NODES[2]);
      int v3 = analogRead(NODES[3]);

      Serial.print(v0); Serial.print(",");
      Serial.print(v1); Serial.print(",");
      Serial.print(v2); Serial.print(",");
      Serial.println(v3);
    }
  }
}