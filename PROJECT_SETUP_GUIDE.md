# Project Setup & Deployment Guide

This guide explains exactly how the files in this project are categorized when moving between computers, servers, or cloned from a new GitHub repository.

---

## 1. What You Will Clone from Git
When you run `git clone`, Git will download the **source code and structure** of the project. These files represent the blueprint of your application but contain NO personal data, large model weights, or private keys.

**Source Code included in Git:**
- **Backend Code:** `backend/` folder (`main.py`, models, routes, logic, html templates, static UI assets).
- **Frontend App:** `flutter_application_1/` folder (all Flutter Dart code, configs).
- **Other UI:** `frontend/` folder (if applicable).
- **Configurations:** `requirements.txt`, `README.md`, `.gitignore`.
- **Environment Template:** `.env.example` (Provides a skeleton for what keys are needed).

---

## 2. What You Must Download (External Dependencies)
These files are too large for Git or are system-specific installed packages. After cloning from Git, you must download/install these separately on the new machine.

**Dependencies to Install:**
- **Python Packages:** Run `pip install -r requirements.txt` to re-download all Python packages (like FastAPI, InsightFace, Torch) into your `.venv` virtual environment.
- **Flutter Packages:** Run `flutter pub get` inside the `flutter_application_1` directory.
- **Node Modules (if applicable):** Run `npm install` in applicable frontend folders.

**AI Model Weights:**
Because deep learning weights are large, they are ignored by Git. You must redownload or transfer them:
- **InsightFace Models:** Download the `buffalo_l` pack (`1k3d68.onnx`, `2d106det.onnx`, `det_10g.onnx`, `genderage.onnx`, `w600k_r50.onnx`) and place them in `backend/insightface_models/models/buffalo_l/`.
- **YOLOv8 Models:** Any `.pt` model files you use (e.g. `yolov8n.pt` or spoofing models) must be downloaded and placed correctly in the root or `backend/` folder depending on your script configuration.

---

## 3. What You Must Transfer Manually (Data & Secrets)
These files are strictly ignored by Git for **Security** and **Privacy** reasons. If you want a new machine to be identical to your current state, you MUST copy these via a secure medium (USB drive, direct file transfer, HTTPS download, SCP).

**Secrets & Keys (CRITICAL):**
- **`.env`**: You must manually create this on the new machine and copy your private variables, database URLs, and JWT secrets from your original `.env` file. Do NOT upload this to Git.
- **SSL Certificates:** `backend/cert.pem` and `backend/key.pem` (Your HTTPS certificates). If setting up on a totally new production server, you MUST generate new ones or copy your existing valid ones.

**Application State & Databases:**
- **SQLite Database:** `attendance.db`. If you want all your registered users, timetables, and past attendance records to migrate, copy this file over.
- **Biometric Profiles (Student Faces):** The `backend/student_faces/` folder containing cropped face images used for recognition must be manually copied. If you don't copy this, the system will not recognize anyone until they re-enroll via mobile or web.
- **Attendance Records (Images):** The `backend/attendance_proofs/` and `backend/class_photos/` folders containing historical CCTV photos and crops. You only copy these if you want to preserve history.
- **Violations:** Logs of spoof attempts under `backend/static/violations/`.

---

## Quick Start Checklist for a New Machine:
1. `git clone <repository_url>`
2. Create and run a new virtual environment:
   - `python -m venv .venv`
   - Activate it (`.venv\Scripts\activate` on Windows)
3. Run `pip install -r requirements.txt`.
4. Manually copy `.env`, `attendance.db`, and `cert.pem/key.pem` from your old machine to the root folder (or `backend/` context depending on where they live).
5. Copy the user DB: `backend/student_faces/`.
6. Download/Copy the AI ONNX/PT models into their respective folders (`backend/insightface_models/models/...`).
7. Run the backend: `cd backend && python main.py` or from root `python backend/main.py` depending on your typical run path.
8. Go to `flutter_application_1/`, run `flutter pub get`, then test via `flutter run`.
