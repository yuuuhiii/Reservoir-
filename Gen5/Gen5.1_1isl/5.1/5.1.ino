/*
 * NARMA10 連続駆動 + 複数タイミング観測 (Arduino Leonardo)
 * 連続駆動を維持したまま1入力内で複数タイミング読み, 次元を増やす.
 */

const int PIN_PWM = 9;
const int NODES[4] = {A0, A1, A2, A3};

const int SAMPLE_OFFSETS_MS[] = {1, 3, 6};   // 4ノード×3タイミング=12次元
const int N_SAMP = sizeof(SAMPLE_OFFSETS_MS) / sizeof(SAMPLE_OFFSETS_MS[0]);

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

      int prev = 0;
      String out = "";
      for (int k = 0; k < N_SAMP; k++) {
        int wait = SAMPLE_OFFSETS_MS[k] - prev;
        if (wait > 0) delay(wait);
        prev = SAMPLE_OFFSETS_MS[k];
        for (int nn = 0; nn < 4; nn++) {
          out += analogRead(NODES[nn]);
          if (!(k == N_SAMP - 1 && nn == 3)) out += ",";
        }
      }
      Serial.println(out);
    }
  }
}