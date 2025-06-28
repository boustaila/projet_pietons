from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import torch
import uuid
import cv2
import numpy as np
import sys

# ✅ YOLOv5 المحلي

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.augmentations import letterbox
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator
import sys
sys.path.append('path/to/yolov5')  # عدل هذا إلى مسار مجلد yolov5 بعد النسخ

# إعداد FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# إعداد المسارات
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "runs/detect"
MODEL_PATH = "models/best.pt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# تحميل النموذج

device = select_device('')
model = DetectMultiBackend(MODEL_PATH, device=device, dnn=False)

# ✅ الصفحة الرئيسية
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ✅ endpoint للكشف
@app.post("/detect")
async def detect(request: Request, file: UploadFile = File(...)):

    def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # width, height padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
        return coords

    # حفظ الصورة
    image_id = str(uuid.uuid4())
    img_path = os.path.join(UPLOAD_FOLDER, f"{image_id}.jpg")
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # تحميل الصورة
    img0 = cv2.imread(img_path)
    img = letterbox(img0, 640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    # تحويل إلى tensor
    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # كشف الأشخاص
    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # رسم النتائج
    annotator = Annotator(img0.copy(), line_width=2)
    for i, det in enumerate(pred):
        if len(det):
            # تحويل الإحداثيات من حجم صورة الإدخال إلى حجم الصورة الأصلية
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = f"{model.names[int(cls)]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=(0, 255, 0))

    img_result = annotator.result()

    # حفظ النتيجة
    result_path = os.path.join(RESULT_FOLDER, f"{image_id}_result.jpg")
    cv2.imwrite(result_path, img_result)

    return FileResponse(result_path, media_type="image/jpeg")
