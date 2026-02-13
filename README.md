# Real-Time Pedestrian Detection with YOLOv5 + FastAPI

This project allows real-time pedestrian detection using a webcam and a trained YOLOv5 model. The app includes a user-friendly **web interface** powered by **FastAPI**, and shows live results with bounding boxes.

##  Project Goal

Detect pedestrians live through your webcam using a modern deep learning model (YOLOv5) and interact with it through a simple HTML interface.

---

##  Technologies Used

-  **YOLOv5** – Object detection (Ultralytics)
-  **FastAPI** – Web API framework
-  **OpenCV** – Webcam video processing
-  **HTML/CSS** – Simple web interface

---


## 📂 Project Structure
```text
├── models/          # Contains the YOLOv5 weights (best.pt)
├── static/          # CSS and images
├── templates/       # HTML files (index.html, live.html)
├── yolov5/          # YOLOv5 core library
├── main.py          # Main application logic
└── requirements.txt # Python dependencies

---

##  How to Run It

### 1.  Clone the repository

```bash
git clone https://github.com/boustaila/projet_pietons.git
cd projet_pietons

### 2. Install dependencies
 pip install -r yolov5/requirements.txt
 pip install fastapi uvicorn opencv-python


Run the server

uvicorn main:app --reload
Access the App Open your browser at http://127.0.0.1:8000

 **Ameed Boustaila**  
Email: boustailaahmed014@gmail.com

