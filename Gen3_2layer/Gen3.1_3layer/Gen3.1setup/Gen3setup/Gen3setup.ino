// reservoir_hil_firmware.ino
const int pwmPin = 9;

void setup() {
  Serial.begin(115200);
  pinMode(pwmPin, OUTPUT);
  // 最初は0Vにしておく
  analogWrite(pwmPin, 0); 
}

void loop() {
  // Pythonからデータ(0〜255)が送られてくるのを待つ
  if (Serial.available() > 0) {
    int inputVal = Serial.parseInt();
    
    // 改行コードなどを読み飛ばす
    if (Serial.read() == '\n') {}

    // 1. リザバーに電圧(波形)を注入
    analogWrite(pwmPin, inputVal);

    // 2. 波が回路内を伝わるのを少し待つ (リザバーの物理的な計算時間)
    delay(10); 

    // 3. 6つのノード(第1層〜第3層)の状態を読み取る
    int e = analogRead(A0);
    int f = analogRead(A1);
    int g = analogRead(A2);
    int h = analogRead(A3);
    int i = analogRead(A4); // Node I (1000μFの深層記憶)
    int j = analogRead(A5); // Node J (カオスフィードバック)

    // 4. Pythonへ結果をカンマ区切りで返す
    Serial.print(e); Serial.print(",");
    Serial.print(f); Serial.print(",");
    Serial.print(g); Serial.print(",");
    Serial.print(h); Serial.print(",");
    Serial.print(i); Serial.print(",");
    Serial.println(j);
  }
}