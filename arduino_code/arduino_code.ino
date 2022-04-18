int x;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {

  while (!Serial.available());
  x = Serial.readString().toInt();
  Serial.print(x);
  if( x==1 ){
    digitalWrite(LED_BUILTIN, HIGH);
    }
  else{
    digitalWrite(LED_BUILTIN, LOW);
    }
}
