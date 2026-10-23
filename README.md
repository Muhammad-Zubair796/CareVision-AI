# 🏥 CareVision AI — Enterprise Hospital Patient Flow & Seating Capacity SaaS

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-purple.svg)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Persistent%20Storage-003B57.svg)](https://www.sqlite.org/)
[![ApexCharts](https://img.shields.io/badge/ApexCharts-Telemetry%20Dashboards-E91E63.svg)](https://apexcharts.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CareVision AI** is an enterprise-grade, edge-compatible AI computer vision and predictive telemetry platform engineered for hospitals, outpatient departments (OPD), emergency rooms (ER), and diagnostic clinics. 

By analyzing existing waiting area CCTV feeds in real time, CareVision AI calculates seat occupancy, identifies standing vs. seated patients using skeletal pose estimation, triggers automated overcrowding alerts to prevent patient walkouts, and logs time-series influx trends into a persistent SQLite database.

---

## 🖥️ Live System Dashboard Preview

### 1. Central Command Overview & Live CCTV Streams
![Central Command Overview & Live CCTV Streams](vision1.PNG)

### 2. Real-Time Telemetry & Peak Surge Analysis
![Live Telemetry Donut & Peak Hours Histogram](vision2.PNG)

### 3. Multi-Day Longitudinal Surge Trends & Ward Telemetry
![30-Day Surge Influx Trend & Telemetry Table](vision3.PNG)

---

## 📑 Table of Contents
1. [Core Features](#-core-features)
2. [System Architecture](#-system-architecture)
3. [Cost & Edge Deployment Strategy](#-cost--edge-deployment-strategy)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Getting Started](#-getting-started)
7. [CCTV & RTSP Camera Integration](#-cctv--rtsp-camera-integration)
8. [Telemetry API Reference](#-telemetry-api-reference)
9. [Configuration & Persistence](#-configuration--persistence)
10. [License](#-license)

---

## 🚀 Core Features

### 1. Computer Vision & Pose Classification
* **Real-time Seated vs. Standing Detection:** Leverages `yolo11n-pose.pt` keypoint tracking (hip, knee, and shoulder spatial vectors) combined with bounding-box aspect ratio classification to accurately determine if a patient is seated or standing.
* **Spatial Seat Overlap Analysis:** Calculates intersection-over-union (IoU) between person keypoint coordinates and chair/bench bounding boxes.
* **Low-Latency Multi-Threaded Streaming:** Uses decoupled `FreshFrameReader` worker threads to capture RTSP/video feeds without buffer lag or memory accumulation.

### 2. Central Command Mission Control
* **Multi-Ward CCTV Matrix:** Live grid preview displaying all departments (ER, OPD, Pediatrics, Terminal Waiting Area) side-by-side with uncropped ultrawide aspect ratio formatting (`1280x520`).
* **Zero-Latency In-Stream HUD Overlay:** Dynamically renders patient counts, seated/standing metrics, chair deficit/surplus status, and real-time administrative action directives directly onto the video composite.
* **One-Click Stream Inspection:** Full-screen inspection modal for detailed observation of any individual camera feed.
* **System Clock:** Synchronized to Pakistan Standard Time (PKT, UTC+5).

### 3. Enterprise Visual Analytics (ApexCharts)
* **Real-time Donut Chart:** Dynamic ratio split visualizer showing Seated Patients, Standing Patients, and Available Chair capacity. Updates every second via polling telemetry.
* **Historical Peak Surge Bar Graph:** Hourly inflow tracker calculating peak bottleneck hours and flagging surges (e.g., `Friday at 09:00 AM (30 Patients)`).
* **Multi-Day Continuous Trend Line Graph:** Smooth spline area curve detailing patient influx wave patterns across selectable historical windows:
  * **24 Hours:** Hourly resolution for the current day.
  * **7 Days:** Daily peak-versus-average flow comparison across the week.
  * **Past 30 Days (1 Month):** Full monthly longitudinal analysis tracking shift surges and calm periods.

### 4. Zero-Friction Settings & Management
* **Department CRUD:** Dynamically add, modify, or delete departments, names, chair capacities, and RTSP video endpoints directly from the UI without restarting the application.
* **State Persistence:** Automatically syncs changes to `hospital_config.json` and records every snapshot into `hospital_analytics.db`.

---

## 🏗 System Architecture

<pre><code>
+-----------------------------------------------------------------------------------+
|                            HOSPITAL LOCAL AREA NETWORK                            |
|                                                                                   |
|  [ CCTV Camera 1 ]     [ CCTV Camera 2 ]     [ CCTV Camera 3 ]                    |
|         │                     │                     │                             |
|         └──────────────┬──────┴─────────────────────┘                             |
|                        ▼ (RTSP Sub-streams / H.264)                               |
|          [ Local Edge Runner / Reception PC ]                                     |
|          - FreshFrameReader (Decoupled capture threads)                           |
|          - YOLO11-Pose Inference (1-2 FPS sampling)                               |
|          - Seated/Standing Pose Classifier                                        |
+────────────────────────┬──────────────────────────────────────────────────────────+
                         │ 
                         │ (Encrypted JSON Telemetry Payload / Zero Video Egress)
                         ▼
+───────────────────────────────────────────────────────────────────────────────────+
|                             CAREVISION CLOUD / ON-PREM                            |
|                                                                                   |
|  [ FastAPI Core Gateway (server.py) ]                                             |
|        ├── settings.py   ──> Hospital Configuration & CRUD (hospital_config.json) |
|        └── analytics.py  ──> SQLite Time-Series DB (hospital_analytics.db)        |
|                                                                                   |
|  [ Central Command Web Interface ]                                                |
|        ├── Real-Time Multi-Cam Grid (Composite HUD stream)                       |
|        ├── Live Aggregate KPI Cards (Patients, Seated, Standing, Empty)           |
|        ├── ApexCharts Live Donut & Peak Hours Timeline Bar Graph                  |
|        └── 30-Day Multi-Day Continuous Spline Area Chart (24h / 7d / 30d filters) |
+-----------------------------------------------------------------------------------+
</code></pre>

---

## 💡 Cost & Edge Deployment Strategy

Streaming 24/7 high-resolution video feeds across the internet to cloud GPU virtual machines is cost-prohibitive for private clinics and mid-sized healthcare facilities. CareVision AI solves this via a **Hybrid Edge Architecture**:

1. **Zero-Hardware-Cost Mode (Existing Reception PC):**
   * The AI runner executes on the clinic's existing Windows desktop located at the front desk or billing counter.
   * By sampling at 1 frame every 2 seconds, CPU utilization remains below 5% on standard Intel Core i3/i5 processors.
   * Video never leaves the clinic's local network (guaranteeing 100% patient privacy and HIPAA compliance).
2. **Predictable Cloud Hosting ($10–$15/month):**
   * The centralized SaaS server only ingests tiny JSON telemetry payloads (less than 1 KB per minute).
   * A single low-tier cloud instance can manage real-time dashboards for over 50 healthcare facilities simultaneously.

---

## 🛠 Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend & Web Server** | FastAPI, Starlette, Uvicorn, Python 3.10+ |
| **Vision & AI Core** | Ultralytics YOLO11 (Pose Estimation), OpenCV |
| **Database & Persistence** | SQLite (`hospital_analytics.db`), JSON File Stores |
| **Frontend & UI** | Tailwind CSS (CDN), Vanilla JavaScript |
| **Telemetry & Visuals** | ApexCharts 3.x (Dark Theme), HTML5 MJPEG Streaming |
| **Timezone Management** | Native Python `datetime` (Asia/Karachi PKT, UTC+5) |

---

## 📁 Project Structure

<pre><code>
CareVision-AI/
├── server.py                 # Main FastAPI application, stream encoder, and command dashboard
├── settings.py               # Settings router, department CRUD, config persistence, and video readers
├── analytics.py              # SQLite analytics engine, 30-day historical data seeder, and trend APIs
├── hospital.py               # Standalone headless vision analytics runner with OpenCV HUD renderer
├── hospital_analytics.db     # Persistent SQLite database storing historical patient telemetry
├── hospital_config.json      # Persistent JSON configuration for departments, chairs, and cameras
├── requirements.txt          # Python dependencies
├── yolo11n-pose.pt           # Lightweight YOLO11 pose estimation model weights
├── yolo11n.pt                # Standard YOLO11 object detection model weights
├── vision1.PNG               # Preview screenshot: Command Overview & Multi-Cam Grid
├── vision2.PNG               # Preview screenshot: Live Ratio Donut & Timeline Bar
├── vision3.PNG               # Preview screenshot: 30-Day Area Chart & Telemetry Table
└── README.md                 # Complete system documentation
</code></pre>

---

## ⚡ Getting Started

### 1. Prerequisites
* Linux (Ubuntu 20.04/22.04 LTS), macOS, or Windows 10/11
* Python 3.10 or 3.11
* Camera RTSP streams or test `.mp4` video clips

### 2. Installation
Clone the repository and configure the virtual environment:

<pre><code>git clone https://github.com/Muhammad-Zubair796/CareVision-AI.git
cd CareVision-AI

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
</code></pre>

### 3. Dependencies (`requirements.txt`)
Ensure your `requirements.txt` contains:

<pre><code>fastapi>=0.110.0
uvicorn[standard]>=0.28.0
ultralytics>=8.3.0
opencv-python-headless>=4.9.0.80
numpy>=1.24.0
python-multipart>=0.0.9
</code></pre>

### 4. Running the Server

#### Option A: Development Mode (Port 8000)
<pre><code>python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
</code></pre>
Access the interface at: `http://localhost:8000`

#### Option B: Production Mode (Port 80)
<pre><code>sudo -E env "PATH=$PATH" python3 -m uvicorn server:app --host 0.0.0.0 --port 80 --workers 1
</code></pre>

#### Option C: Run in Background (Daemonized with nohup)
<pre><code>sudo -E env "PATH=$PATH" nohup python3 -m uvicorn server:app --host 0.0.0.0 --port 80 --workers 1 > server.log 2>&1 &
</code></pre>

---

## 📹 CCTV & RTSP Camera Integration

CareVision AI ingests standard RTSP streams provided by all commercial Network Video Recorders (NVR) and IP cameras.

### RTSP URI Reference Guide:
* **Hikvision:** `rtsp://admin:password@192.168.1.100:554/Streaming/Channels/102` *(102 indicates Sub-stream)*
* **Dahua:** `rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=1`
* **Uniview (UNV):** `rtsp://admin:password@192.168.1.100:554/unicast/c1/s1/live`
* **Generic ONVIF:** `rtsp://admin:password@192.168.1.100:554/ch01_sub.264`

> **Optimization Tip:** Always configure cameras to provide their **Sub-Stream** (typically 640x360 or 720p). YOLO processes images at 640px resolution; utilizing sub-streams eliminates unnecessary CPU decoding overhead without degrading detection accuracy.

---

## 📡 Telemetry API Reference

| Endpoint | Method | Description | Sample Output |
| :--- | :--- | :--- | :--- |
| `/api/central_stats` | `GET` | Fetches real-time aggregate KPI counters, live ward counts, and clock. | `<pre><code>{"tot_p": 16, "tot_s": 12, "tot_st": 4, "tot_e": 51}</code></pre>` |
| `/api/analytics_history`| `GET` | Returns chronological telemetry buckets and peak bottleneck indicators for the bar chart. | `<pre><code>{"categories": [...], "totals": [...], "peak_info": "..."}</code></pre>` |
| `/api/trend_data` | `GET` | Returns multi-day influx arrays filtered by range (`24h`, `7d`, `30d`). | `<pre><code>{"categories": [...], "totals": [...], "seated": [...]}</code></pre>` |
| `/stream/{dept_id}` | `GET` | Multipart MJPEG real-time video stream with annotated HUD overlay. | `multipart/x-mixed-replace` |

---

## ⚙️ Configuration & Persistence

All hospital configurations and metrics are stored locally:

* **`hospital_config.json`**: Preserves hospital identification, ward names, chair allocations, and RTSP connection strings across reboots.
* **`hospital_analytics.db`**: SQLite database storing timestamped patient count snapshots. Can be directly queried via standard SQL tools:

<pre><code>sqlite3 hospital_analytics.db "SELECT date_str, time_str, total_patients, seated, standing FROM patient_snapshots ORDER BY id DESC LIMIT 10;"
</code></pre>

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
