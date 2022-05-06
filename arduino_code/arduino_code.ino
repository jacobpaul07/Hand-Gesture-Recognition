int x;
int relay_one = 5 ;  
int relay_two = 6;
int relay_three = 7;
int relay_four = 8;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(relay_one, OUTPUT);
  pinMode(relay_two, OUTPUT);
  pinMode(relay_three, OUTPUT);
  pinMode(relay_four, OUTPUT);
}

void loop() {

  while (!Serial.available());
  x = Serial.readString().toInt();
  Serial.print(x);
  if(x==111){
    digitalWrite(relay_one, HIGH);
    }
  else if(x==110){
    digitalWrite(relay_one, LOW);
    }
  else if(x==121){
    digitalWrite(relay_two, HIGH);
    }
  else if(x==120){
    digitalWrite(relay_two, LOW);
    }
  else if(x==211){
    digitalWrite(relay_three, HIGH);
    }
  else if(x==210){
    digitalWrite(relay_three, LOW);
    }
  else if(x==221){
    digitalWrite(relay_four, HIGH);
    }
  else if(x==220){
    digitalWrite(relay_four, LOW);
    }
}
