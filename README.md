# 🐦 Bird Species Detection using YOLOv8

*A beginner-friendly deep learning project — my first custom object detection model!*

This repository contains my first end-to-end **custom object detection project**, where I trained a YOLOv8 model to detect **four bird species**:

* **Pigeon**
* **Hen**
* **Owl**
* **Eagle**

Even though this is my first ML/DL project, I successfully collected a dataset, labeled images, trained a YOLO model locally on GPU, and built an inference script to test images, folders, videos, and webcam streams.

This may not be the highest-accuracy model, but it marks the **first step of my machine learning journey** 🚀.

---

## 📁 Project Structure

```
object-detection-project/
│
├── 1_datapreparation/
│   └── data_images/
│       ├── train/          # Training images
│       ├── test/           # Validation images
│       └── Annotation/     # YOLO label files
│
├── data.yaml               # YOLO dataset configuration
├── run_inference.py        # Script to run the model on images/videos
│
└── runs/
    └── detect/
        └── trainX/         # YOLO training outputs (best.pt, results)
```

---

## 🧠 Model & Training

I used **YOLOv8n (nano)** — a lightweight model suitable for beginners.

Training was done locally using:

* Python **3.13**
* PyTorch **CUDA 12.9 build**
* GPU: **NVIDIA RTX 3050 6GB**

Training command:

```
yolo detect train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640 device=0
```

After training, YOLO outputs:

```
runs/detect/trainX/weights/best.pt  
runs/detect/trainX/weights/last.pt
```

**best.pt** is used for inference.

---

## 📈 Model Performance

On a small dataset (~200 images):

| Metric   | Score |
| -------- | ----- |
| mAP50    | ~95%  |
| mAP50-95 | ~78%  |

Great results considering limited data and being the **first model I've trained**.

---

## ▶️ Inference (Run the Model)

Use:

```
python run_inference.py --source <path>
```

### Examples

Run on single image:

```
python run_inference.py --source data_images/test/31.jpg
```

Run on a folder:

```
python run_inference.py --source myfolder/
```

Run on a video:

```
python run_inference.py --source video.mp4
```

Run webcam:

```
python run_inference.py --source 0
```

The window stays open until you press **Q**.

---

## 🙌 Learnings From This Project

This project helped me understand:

* Dataset preparation and image annotation
* YOLO bounding box format
* Training models locally with GPU
* Managing virtual environments
* Fixing CUDA & PyTorch issues
* Writing custom inference scripts
* Structuring a production-like ML project

This may be a simple project, but it represents **real progress** and hands-on ML experience.

---

## 🚀 Future Improvements

* Train on a larger dataset
* Add additional bird species
* Use a larger backbone (YOLOv8m / v8l)
* Deploy using Streamlit or FastAPI
* Convert to ONNX or TensorRT

---

## 📜 License

This project is free to use for learning and educational purposes.
