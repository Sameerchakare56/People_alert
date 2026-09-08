# 🎥 People Alert - Real-Time Person Tracking & Boundary Crossing Alert System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, high-performance **Computer Vision & AI Video Analytics** application designed for real-time person tracking, directional virtual boundary line monitoring, and instant alert notifications. Built with **YOLOv8**, **OpenCV**, **FastAPI**, and a **Glassmorphic Web Control Center UI**.

---

## 🌟 Key Features

- **🤖 YOLOv8 Person Detection & Centroid Tracking**: Tracks unique human targets across video frames with unique IDs and real-time motion trajectory paths.
- **📐 Interactive Angled Virtual Boundary Line**: Dynamic line equation calculation allowing 0° to 360° line rotation and 0% to 100% perpendicular position shifts.
- **🎯 Directional Crossing Detection**: Configurable alert triggers for **Left-to-Right**, **Right-to-Left**, or **Both Sides** boundary line crossings.
- **📤 Custom Video File Upload**: Drag-and-drop or select MP4, AVI, MOV, MKV, or WEBM video files directly in the browser to analyze custom video footage.
- **📡 Multi-Source Support**: Connect live RTSP camera feeds, webcams, or default local video streams smoothly.
- **⚡ Thread-Safe Video Capture**: Background thread locks (`cap_lock`) prevent video feed interruption when switching sources or uploading files.
- **🚨 Instant Alerts & Snapshots**: Generates real-time cropped snapshot images of boundary crossers with timestamps and audio alerts.
- **🎨 Glassmorphic Control Center UI**: Modern dark-mode web dashboard with real-time statistics, compass visualizer, and alert card grid.

---

## 🏗️ System Architecture

```
                                  +----------------------------------+
                                  |       Web Dashboard UI           |
                                  |    (index.html - Dark Glass)    |
                                  +----------------+-----------------+
                                                   |
                               HTTP GET /video_feed| REST API / Upload
                                (MJPEG Streaming)  | Config & Controls
                                                   v
+-----------------------+         +----------------+-----------------+
|   Video Input Source  |         |        FastAPI Backend           |
| (Uploaded File / RTSP |-------->|        (det-alert.py)          |
|   Webcam / vi.mp4)    |         +----------------+-----------------+
+-----------------------+                          |
                                                   v
                                  +----------------+-----------------+
                                  |      AI Processing Pipeline      |
                                  |  - YOLOv8 Object Detection       |
                                  |  - Centroid Person Tracker       |
                                  |  - Line Distance Math Engine     |
                                  +----------------+-----------------+
```

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn, PyTorch
- **Computer Vision**: OpenCV (`cv2`), Ultralytics YOLOv8, NumPy
- **Frontend**: HTML5, Vanilla JavaScript (ES6+), CSS3 (Glassmorphic Design System)
- **Dependencies**: `python-multipart`, `requests`, `Pillow`

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/People_alert.git
cd People_alert
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python det-alert.py
```

### 5. Open Web Control Dashboard
Open your web browser and navigate to:
```text
http://localhost:8000
```
*(Or simply open `index.html` directly in your browser)*

---

## 📡 REST API Documentation

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/video_feed` | `GET` | Returns real-time MJPEG video stream with bounding boxes and boundary line overlays. |
| `/upload_video` | `POST` | Uploads a custom video file (`UploadFile`) and switches active stream source. |
| `/config/video_source` | `POST` | Switches active video source (`local`, `rtsp`, or `webcam`). |
| `/config/boundary` | `POST` | Updates boundary ratio (0.0-1.0), angle (0-360°), and detection side (`left`, `right`, `both`). |
| `/stats` | `GET` | Returns active tracked targets count, alert count, line angle, and source info. |
| `/latest_crossing` | `GET` | Returns latest alert event payload with base64 encoded snapshot image. |
| `/reset` | `POST` | Resets all active tracking IDs and alert counters. |

---

## 📂 Project Structure

```text
People_alert/
├── det-alert.py          # Core FastAPI Application & Computer Vision Pipeline
├── index.html            # Ultra-Modern Web Dashboard Interface
├── requirements.txt      # Python Dependencies Manifest
├── yolov8n.pt            # Pre-trained YOLOv8 Nano Model Weights
├── vi.mp4                # Default Sample Video
├── uploads/              # Directory for User-Uploaded Video Files
└── README.md             # Project Documentation
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/YourUsername/People_alert/issues).

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
