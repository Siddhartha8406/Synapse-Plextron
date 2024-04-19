#include<servo.h>

Servo Axis1;
Servo Axis2;

void setup() {
  Axis1.attach(9);
  Axis1.attach(10);

  Axis1.write(0);
  Axis1.write(90);

  Serial.begin(9600);
  Serial.println("Start")
}

void loop() {
  // put your main code here, to run repeatedly:

}
