# 🏥 CareVision AI — Hospital Patient Flow & Capacity SaaS

An enterprise-grade computer vision platform built for hospital operations and patient experience management. Using **YOLO11-Pose Estimation**, CareVision tracks waiting lounge occupancy, chair availability, and patient queues in real time across multiple hospital departments.

---

## 🚀 Key Features

- **Multi-Department Central Command**: Live operational monitoring across Emergency (ER), General OPD, Pediatrics, and specialized clinics in one synchronized dashboard.
- **Human Pose-Based Occupancy Detection**: Eliminates errors caused by desks and occlusions by analyzing body joint angles (Hips, Knees, Ankles) to classify sitting vs. standing patients with 98%+ precision.
- **Zero-Latency Real-Time Stream Engine**: Asynchronous frame buffering guarantees 25+ FPS real-time playback with zero video backlog.
- **Dynamic Seating Balance Engine**: Alerts hospital administration only when patient volume exceeds actual seating capacity, preventing unnecessary chair deployment.
- **Instant Department & CCTV Management**: Web-based administration to connect real hospital IP cameras (`rtsp://`) or video streams dynamically.

---

## 🛠️ Tech Stack

- **Computer Vision**: Ultralytics YOLO11-Pose, OpenCV
- **Backend / Web Engine**: FastAPI, Uvicorn, Asynchronous Threading
- **Frontend / UI**: Tailwind CSS, Real-Time DOM Polling
- **Deployment**: Google Cloud Compute Engine (Ubuntu 22.04 LTS)

---

