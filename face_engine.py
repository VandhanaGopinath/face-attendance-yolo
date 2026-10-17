"""
face_engine.py - the face pipeline shared by every script.

    frame --YOLOv8-face--> face box + 5 landmarks
          --align-------->  112x112 upright face
          --ArcFace------>  512-number "face embedding" (unit length)
          --cosine------->  who is this? (or "Unknown")

Why each step exists
    * YOLO face detector  - a CNN object detector (single forward pass over the whole
      image). It returns a box, a confidence score and 5 landmarks (eyes, nose, mouth corners).
    * Alignment           - the 5 landmarks are used to rotate/scale/shift the face onto a
      standard template, so every face reaches the recogniser in the same pose and size.
    * ArcFace embedding   - a MobileFaceNet network trained with the ArcFace loss turns a face
      into 512 numbers. Photos of the same person land close together, different people far apart.
    * Cosine similarity   - since embeddings have length 1, similarity is just a dot product.
      The best score above a threshold is the identity; below it the face is "Unknown".
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DETECTOR_PATH = os.path.join(MODELS_DIR, "yolov8n-face.pt")
RECOGNIZER_PATH = os.path.join(MODELS_DIR, "w600k_mbf.onnx")
GALLERY_PATH = os.path.join(BASE_DIR, "data", "gallery.npz")

# Where the five landmarks (left eye, right eye, nose, left mouth, right mouth) must land
# in a 112x112 ArcFace input. These are the standard ArcFace template coordinates.
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

DEFAULT_THRESHOLD = 0.40   # cosine similarity needed to accept an identity (tune with evaluate.py)


@dataclass
class Face:
    box: tuple          # (x1, y1, x2, y2) in pixels
    conf: float         # detector confidence, 0..1
    landmarks: np.ndarray  # (5, 2)
    embedding: np.ndarray = None  # (512,), unit length


class FaceEngine:
    def __init__(self, det_path=DETECTOR_PATH, rec_path=RECOGNIZER_PATH, det_conf=0.5, det_size=640):
        for path in (det_path, rec_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Model file not found: {path}\nRun:  python download_models.py")
        # imported here so that scripts that only need the Gallery start quickly
        from ultralytics import YOLO, settings
        import onnxruntime as ort

        try:
            settings.update({"sync": False})   # no usage-analytics calls
        except Exception:
            pass
        self.detector = YOLO(det_path)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3          # hide a harmless "batch size" warning
        self.recognizer = ort.InferenceSession(rec_path, sess_options=opts,
                                               providers=["CPUExecutionProvider"])
        self._rec_input = self.recognizer.get_inputs()[0].name
        self.det_conf = det_conf
        self.det_size = det_size

    # ---- 1. detection ------------------------------------------------------
    def detect(self, frame):
        """Return a list of Face (no embedding yet), largest face first."""
        result = self.detector.predict(frame, conf=self.det_conf, imgsz=self.det_size, verbose=False)[0]
        faces = []
        if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
            return faces
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        kpts = result.keypoints.xy.cpu().numpy()
        for box, conf, kp in zip(boxes, confs, kpts):
            faces.append(Face(tuple(float(v) for v in box), float(conf), kp.astype(np.float32)))
        faces.sort(key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]), reverse=True)
        return faces

    # ---- 2. alignment ------------------------------------------------------
    @staticmethod
    def align(frame, landmarks):
        """Warp the face so its landmarks sit on the ArcFace template (112x112)."""
        matrix, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_TEMPLATE, method=cv2.LMEDS)
        if matrix is None:
            return None
        return cv2.warpAffine(frame, matrix, (112, 112), borderValue=0.0)

    # ---- 3. embedding ------------------------------------------------------
    def embed_aligned(self, aligned_faces):
        """List of 112x112 BGR faces -> (N, 512) array of unit-length embeddings."""
        blob = cv2.dnn.blobFromImages(aligned_faces, 1.0 / 127.5, (112, 112),
                                      (127.5, 127.5, 127.5), swapRB=True)
        out = self.recognizer.run(None, {self._rec_input: blob})[0]
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def analyse(self, frame, min_face_px=0):
        """Detect + align + embed every face in a frame. Returns a list of Face with .embedding."""
        faces = [f for f in self.detect(frame) if (f.box[2] - f.box[0]) >= min_face_px]
        aligned, keep = [], []
        for f in faces:
            a = self.align(frame, f.landmarks)
            if a is not None:
                aligned.append(a)
                keep.append(f)
        if keep:
            for f, emb in zip(keep, self.embed_aligned(aligned)):
                f.embedding = emb
        return keep


class Gallery:
    """The registered people: one stored embedding per enrolment photo/frame."""

    def __init__(self, names=None, embeddings=None):
        self.names = list(names) if names is not None else []
        self.embeddings = (np.asarray(embeddings, dtype=np.float32)
                           if embeddings is not None else np.zeros((0, 512), dtype=np.float32))
        self._centroids = None

    @classmethod
    def load(cls, path=GALLERY_PATH):
        if not os.path.isfile(path):
            return cls()
        data = np.load(path, allow_pickle=False)
        return cls(data["names"].tolist(), data["embeddings"])

    def save(self, path=GALLERY_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, names=np.array(self.names), embeddings=self.embeddings)

    @property
    def people(self):
        return sorted(set(self.names))

    def add(self, name, embeddings):
        """Register a person. If the name already exists, their old samples are replaced."""
        embeddings = np.asarray(embeddings, dtype=np.float32)
        keep = [i for i, n in enumerate(self.names) if n != name]
        self.names = [self.names[i] for i in keep] + [name] * len(embeddings)
        self.embeddings = np.vstack([self.embeddings[keep], embeddings]) if keep else embeddings
        self._centroids = None

    def _build_centroids(self):
        people = self.people
        cents = []
        for p in people:
            idx = [i for i, n in enumerate(self.names) if n == p]
            c = self.embeddings[idx].mean(axis=0)
            cents.append(c / np.linalg.norm(c))
        self._centroids = (people, np.vstack(cents))

    def identify(self, embedding, threshold=DEFAULT_THRESHOLD):
        """Return (name or None, best_score). Compares against each person's mean embedding."""
        if not self.names:
            return None, 0.0
        if self._centroids is None:
            self._build_centroids()
        people, cents = self._centroids
        scores = cents @ embedding          # cosine similarity (all vectors are unit length)
        best = int(np.argmax(scores))
        score = float(scores[best])
        return (people[best] if score >= threshold else None), score
