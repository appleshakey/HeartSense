#define ECG_PIN A0  // The analog pin connected to the AD8228 output

void setup() {
  Serial.begin(115200);  // Use the same baud rate in Python
  pinMode(ECG_PIN, INPUT);
}

void loop() {
  int ecg_value = analogRead(ECG_PIN);  // Read the analog value from the sensor
  Serial.println(ecg_value);  // Send the data to the serial port
  delay(4);  // Adjust delay as needed to control the sampling rate
}
