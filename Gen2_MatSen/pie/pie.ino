void setup() {
  // PCとの通信速度を最大に設定
  Serial.begin(115200);
}

void loop() {
  // 8つのノードの電圧を最速で読み取る
  // ※Leonardo/Microの場合、A6はD4ピン、A7はD6ピンになります
  int node0 = analogRead(A0);
  int node1 = analogRead(A1);
  int node2 = analogRead(A2);
  int node3 = analogRead(A3);
  int node4 = analogRead(A4);
  int node5 = analogRead(A5);
  int node6 = analogRead(A6); 
  int node7 = analogRead(A7); 

  // Python側のコード変更を最小限にするため、頭にダミーの "Orig:0," を付ける
  Serial.print("Orig:0,"); 
  Serial.print("N0:"); Serial.print(node0); Serial.print(",");
  Serial.print("N1:"); Serial.print(node1); Serial.print(",");
  Serial.print("N2:"); Serial.print(node2); Serial.print(",");
  Serial.print("N3:"); Serial.print(node3); Serial.print(",");
  Serial.print("N4:"); Serial.print(node4); Serial.print(",");
  Serial.print("N5:"); Serial.print(node5); Serial.print(",");
  Serial.print("N6:"); Serial.print(node6); Serial.print(",");
  Serial.print("N7:"); Serial.println(node7);

  // サンプリングレートを稼ぐための極小ディレイ（波形が細かすぎる場合は5などに増やしてください）
  delay(2); 
}