import cv2
from ultralytics import YOLO
import argparse
import os

p = argparse.ArgumentParser()
p.add_argument("--source", type=str, required=True)
p.add_argument("--model", type=str, default="runs/detect/train4/weights/best.pt")
p.add_argument("--scale", type=float, default=0.4)
p.add_argument("--conf", type=float, default=0.25)
args = p.parse_args()

model = YOLO(args.model)

def show_image(img):
    h, w = img.shape[:2]
    img_small = cv2.resize(img, (int(w * args.scale), int(h * args.scale)))
    cv2.imshow("Result", img_small)

src = args.source

if os.path.isdir(src):
    files = [os.path.join(src, f) for f in os.listdir(src) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
    files.sort()
    for fp in files:
        img = cv2.imread(fp)
        if img is None:
            continue
        results = model(img, conf=args.conf)
        annotated = results[0].plot()
        show_image(annotated)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    exit()

# try opening as video/camera
cap = cv2.VideoCapture(src)
is_video = False
if cap.isOpened():
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frames > 1:
        is_video = True
    else:
        # single-frame video capture (some image paths may make cap.isOpened True)
        is_video = False
else:
    cap.release()
    cap = None

if not is_video and (cap is None):
    img = cv2.imread(src)
    if img is None:
        print("Failed to open source:", src)
        exit(1)
    results = model(img, conf=args.conf)
    annotated = results[0].plot()
    show_image(annotated)
    while True:
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    exit()

if is_video:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=args.conf)
        annotated = results[0].plot()
        show_image(annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    # cap was opened but treated as single image
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("Failed to read frame.")
        exit(1)
    results = model(frame, conf=args.conf)
    annotated = results[0].plot()
    show_image(annotated)
    while True:
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
