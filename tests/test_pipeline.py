"""
Quick offline self-check (about 10 seconds, no camera, no photos needed):

    python tests/test_pipeline.py

It checks the parts that do not need a real face: the gallery maths, face alignment, that both
model files load and behave, and the attendance CSV logic. For recognition quality on real faces
use evaluate.py with your own photos.
"""

import os
import sys
import tempfile
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

import attendance  # noqa: E402
from face_engine import ARCFACE_TEMPLATE, FaceEngine, Gallery  # noqa: E402

checks = []


def check(name, condition):
    checks.append(condition)
    print(("PASS  " if condition else "FAIL  ") + name)


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


# ---- Gallery -------------------------------------------------------------------------
rng = np.random.default_rng(0)
anna, ben = unit(rng.normal(size=512)), unit(rng.normal(size=512))
noisy = lambda base: np.vstack([unit(base + 0.03 * rng.normal(size=512)) for _ in range(5)])

g = Gallery()
g.add("Anna", noisy(anna))
g.add("Ben", noisy(ben))
check("gallery lists both people", g.people == ["Anna", "Ben"])
check("recognises Anna", g.identify(unit(anna + 0.03 * rng.normal(size=512)), 0.4)[0] == "Anna")
check("recognises Ben", g.identify(unit(ben + 0.03 * rng.normal(size=512)), 0.4)[0] == "Ben")
stranger = unit(rng.normal(size=512))
check("a stranger is Unknown", g.identify(stranger, 0.4)[0] is None)
g.add("Anna", noisy(anna))
check("re-registering replaces, does not duplicate", g.names.count("Anna") == 5 and len(g.names) == 10)
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "gallery.npz")
    g.save(path)
    g2 = Gallery.load(path)
    check("gallery survives save/load", g2.people == g.people and np.allclose(g2.embeddings, g.embeddings))
check("empty gallery identifies nobody", Gallery().identify(anna)[0] is None)

# ---- Alignment -----------------------------------------------------------------------
angle, scale, shift = 20.0, 0.5, np.array([300.0, 150.0])
rot = cv2.getRotationMatrix2D((0, 0), -angle, 1.0 / scale)[:, :2]
landmarks = (ARCFACE_TEMPLATE @ rot.T + shift).astype(np.float32)   # the template, tilted/enlarged/moved
frame = np.zeros((600, 800, 3), np.uint8)
aligned = FaceEngine.align(frame, landmarks)
check("alignment returns a 112x112 image", aligned is not None and aligned.shape == (112, 112, 3))
m, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_TEMPLATE, method=cv2.LMEDS)
mapped = landmarks @ m[:, :2].T + m[:, 2]
check("landmarks land on the ArcFace template", np.abs(mapped - ARCFACE_TEMPLATE).max() < 0.5)

# ---- Models --------------------------------------------------------------------------
engine = FaceEngine()
noise_a = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
noise_b = rng.integers(0, 255, (112, 112, 3), dtype=np.uint8)
emb = engine.embed_aligned([noise_a, noise_b])
check("embeddings are 512-dim", emb.shape == (2, 512))
check("embeddings have length 1", np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4))
check("same input gives same embedding", np.allclose(engine.embed_aligned([noise_a])[0], emb[0], atol=1e-4))
check("different inputs give different embeddings", float(emb[0] @ emb[1]) < 0.95)
check("blank image has no faces", engine.analyse(np.zeros((480, 640, 3), np.uint8)) == [])

# ---- Attendance CSV ------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    attendance.ATTENDANCE_DIR = tmp
    now = datetime.now()
    path = attendance.attendance_file(now)
    check("nobody marked on a fresh day", attendance.already_marked_today(path) == set())
    attendance.mark(path, "Anna", 0.91, now)
    attendance.mark(path, "Ben", 0.77, now)
    check("marked people are read back (header ignored)", attendance.already_marked_today(path) == {"Anna", "Ben"})
    with open(path) as f:
        check("csv has header + 2 rows", len(f.read().strip().splitlines()) == 3)

print(f"\n{sum(checks)}/{len(checks)} checks passed")
sys.exit(0 if all(checks) else 1)
