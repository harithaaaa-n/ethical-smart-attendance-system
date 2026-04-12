#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>

void setup() {
  Serial.begin(115200);

  BLEDevice::init("CSE_A_101");   // Classroom ID

  BLEServer *pServer = BLEDevice::createServer();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();

  BLEAdvertisementData advData;
  advData.setName("CYBER_2");   // Beacon name
  advData.setManufacturerData("ATTENDANCE_SYSTEM");

  pAdvertising->setAdvertisementData(advData);
  pAdvertising->setScanResponse(true);   // 🔥 Improves detection

  pAdvertising->start();

  Serial.println("BLE Beacon Started Successfully...");
}

void loop() {
  delay(200);
}