"""
download_models.py - fetch the two model files into ./models if they are missing.

    python download_models.py

  yolov8n-face.pt  - YOLOv8-nano face detector with 5 landmarks (~6 MB)
                     from https://github.com/akanametov/yolo-face
  w600k_mbf.onnx   - ArcFace MobileFaceNet recogniser (~14 MB), taken from InsightFace's
                     "buffalo_s" pack (a 127 MB zip; only this one file is kept)
"""

import io
import os
import urllib.request
import zipfile

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
YOLO_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"


def download(url):
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    yolo_path = os.path.join(MODELS_DIR, "yolov8n-face.pt")
    if os.path.isfile(yolo_path):
        print("yolov8n-face.pt already present")
    else:
        with open(yolo_path, "wb") as f:
            f.write(download(YOLO_URL))
        print("saved", yolo_path)

    arc_path = os.path.join(MODELS_DIR, "w600k_mbf.onnx")
    if os.path.isfile(arc_path):
        print("w600k_mbf.onnx already present")
    else:
        archive = zipfile.ZipFile(io.BytesIO(download(BUFFALO_URL)))
        with open(arc_path, "wb") as f:
            f.write(archive.read("w600k_mbf.onnx"))
        print("saved", arc_path)


if __name__ == "__main__":
    main()
