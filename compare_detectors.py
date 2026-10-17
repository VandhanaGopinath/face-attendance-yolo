"""
compare_detectors.py - old detector (Haar Cascade) vs the new one (YOLOv8-face), side by side.

Give it a folder of photos that each contain exactly ONE face (for example your registration
photos). It then repeats the test under harder conditions - dim light, tilted head, small face,
blur - and reports, for each detector:
    * detection rate  - photos where a face was found
    * false boxes     - photos where MORE than one box was drawn (there is only one face!)
    * speed           - milliseconds per image on this machine

    python compare_detectors.py --data data/eval/Safina
"""

import argparse
import os
import time

import cv2
import numpy as np

from face_engine import FaceEngine

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def dim(img):
    return np.clip(img.astype(np.float32) * 0.25, 0, 255).astype(np.uint8)


def tilt(img):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 25, 1.0)
    return cv2.warpAffine(img, m, (w, h))


def small(img):
    """Face far from the camera: shrink the photo and pad it onto a larger canvas."""
    h, w = img.shape[:2]
    s = cv2.resize(img, (w // 4, h // 4))
    canvas = np.zeros_like(img)
    canvas[: s.shape[0], : s.shape[1]] = s
    return canvas


def blur(img):
    return cv2.GaussianBlur(img, (0, 0), 6)


CONDITIONS = {"normal": lambda x: x, "dim light": dim, "tilted 25 deg": tilt,
              "small face": small, "blurred": blur}


def haar_detect(cascade, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return len(cascade.detectMultiScale(gray, 1.3, 5))      # same settings as the original project


def main():
    parser = argparse.ArgumentParser(description="Haar Cascade vs YOLOv8-face")
    parser.add_argument("--data", required=True, help="folder of photos, each with exactly one face")
    args = parser.parse_args()

    files = [os.path.join(args.data, f) for f in sorted(os.listdir(args.data)) if f.lower().endswith(IMAGE_EXTS)]
    images = [im for im in (cv2.imread(f) for f in files) if im is not None]
    if not images:
        raise SystemExit("No images found.")
    # keep the test fast and fair: limit the long side to 960 px
    images = [cv2.resize(im, None, fx=960 / max(im.shape[:2]), fy=960 / max(im.shape[:2]))
              if max(im.shape[:2]) > 960 else im for im in images]

    cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    engine = FaceEngine()
    engine.detect(images[0])                                  # warm-up

    print(f"{len(images)} photos, one face each\n")
    print(f"{'condition':14} | {'Haar found':>10} {'Haar extra':>10} | {'YOLO found':>10} {'YOLO extra':>10}")
    print("-" * 66)
    totals = {"haar": 0.0, "yolo": 0.0}
    for cond, fn in CONDITIONS.items():
        h_found = h_extra = y_found = y_extra = 0
        for im in images:
            test = fn(im)
            t = time.perf_counter(); n = haar_detect(cascade, test); totals["haar"] += time.perf_counter() - t
            h_found += n >= 1; h_extra += n > 1
            t = time.perf_counter(); n = len(engine.detect(test)); totals["yolo"] += time.perf_counter() - t
            y_found += n >= 1; y_extra += n > 1
        N = len(images)
        print(f"{cond:14} | {h_found / N * 100:9.0f}% {h_extra / N * 100:9.0f}% | "
              f"{y_found / N * 100:9.0f}% {y_extra / N * 100:9.0f}%")
    calls = len(images) * len(CONDITIONS)
    print(f"\nSpeed (CPU): Haar {totals['haar'] / calls * 1000:.0f} ms/image, YOLO {totals['yolo'] / calls * 1000:.0f} ms/image")
    print("'extra' = photos where more than one box was drawn although there is only one face.")


if __name__ == "__main__":
    main()
