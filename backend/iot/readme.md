# 📡 IoT Module – ESP32 BLE Attendance Verification

## 📌 Overview

This module contains the ESP32 (IoT) implementation used for **Bluetooth Low Energy (BLE) based proximity verification** in the Smart Attendance System.

The ESP32 acts as a **BLE Beacon**, continuously broadcasting a unique classroom identifier. The mobile application scans for this signal to ensure that a student is physically present within the classroom before marking attendance.

---

## 🎯 Purpose

The main objective of this module is to:

* Verify **physical presence** of students
* Prevent **proxy attendance**
* Add an extra layer of **security** to AI-based facial recognition

---

## ⚙️ How It Works

1. ESP32 initializes Bluetooth in BLE mode
2. It broadcasts a **classroom identifier (UUID / Name)**
3. The mobile app scans for nearby BLE signals
4. If the signal is detected within range:

   * Student is considered **present in classroom**
5. Backend system allows attendance marking only after:

   * Face recognition ✅
   * BLE proximity verification ✅

---

## 📁 File Structure

```
iot/
 ├── esp32_ble_beacon.ino
 └── README.md
```

---

## 💻 ESP32 Code Description

* **BLEDevice::init()** → Initializes BLE with classroom ID
* **BLEAdvertising** → Enables broadcasting
* **setAdvertisementData()** → Sets beacon name and data
* **start()** → Starts BLE signal transmission

---

## 🔧 Requirements

* ESP32 Development Board
* Arduino IDE
* ESP32 Board Package installed
* USB Cable for uploading code

---

## 🚀 Setup Instructions

1. Open the `.ino` file in Arduino IDE
2. Select Board → ESP32 Dev Module
3. Connect ESP32 via USB
4. Upload the code
5. Open Serial Monitor to verify output

---

## 📡 BLE Configuration

* **Device Name:** CSE_A_101
* **Beacon Name:** CYBER_2
* **Broadcast Data:** ATTENDANCE

---

## 🔐 Security & Ethical Consideration

* Ensures **real-world presence validation**
* Prevents **remote attendance fraud**
* Works alongside encrypted facial recognition system
* Supports **ethical AI principles (fairness & accountability)**

---

## 💡 Future Enhancements

* RSSI-based distance filtering
* Multi-classroom beacon support
* Dynamic beacon ID configuration
* Integration with cloud-based IoT monitoring

---

## 🧠 Summary

This IoT module acts as a **physical verification layer** in the attendance system, ensuring that attendance is marked only when the student is **present in the classroom**, making the system more secure and reliable.
