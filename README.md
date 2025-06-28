# 🚶‍♂️ Real-Time Pedestrian Detection with YOLOv5 + FastAPI

This project allows real-time pedestrian detection using a webcam and a trained YOLOv5 model. The app includes a user-friendly **web interface** powered by **FastAPI**, and shows live results with bounding boxes.

## 🎯 Project Goal

Detect pedestrians live through your webcam using a modern deep learning model (YOLOv5) and interact with it through a simple HTML interface.

---

## 🛠️ Technologies Used

- 🧠 **YOLOv5** – Object detection (Ultralytics)
- ⚡ **FastAPI** – Web API framework
- 🎥 **OpenCV** – Webcam video processing
- 🖼️ **HTML/CSS** – Simple web interface

---

## 📁 Project Structure
- `main.py`: Runs the FastAPI server
- `templates/`: Contains the web UI (index.html)
- `static/`: Custom CSS for styling
- `uploads/`: Images uploaded by user
- `runs/detect/`: Detection result images
- `models/best.pt`: Trained YOLOv5 weights

---

## 🚀 How to Run It

### 1. 🔁 Clone the repository

```bash
git clone https://github.com/boustaila/projet_pietons.git
cd projet_pietons

### 2. 📦 Install dependencies
 pip install -r yolov5/requirements.txt
 pip install fastapi uvicorn opencv-python
### 3. 🚀 Launch the FastAPI server 
 `python main.py`
### 4. Open browser: 
 `http://localhost:8000`

 **Ameed Boustaila**  
Email: boustailaahmed014@gmail.com

