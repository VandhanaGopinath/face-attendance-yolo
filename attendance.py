"""
attendance.py - live recognition + automatic attendance.

    python attendance.py                     # webcam
    python attendance.py --source clip.mp4   # or a video file

For every frame: YOLO finds the faces, each face is aligned and turned into an embedding,
and the closest registered person (cosine similarity) is shown on the box. If the best score
is below --threshold the face is shown as "Unknown".

A person is marked present automatically once they have been recognised in --confirm
consecutive frames (so one lucky frame can't mark someone), and only once per day.
Rows go to Attendance/Attendance_<DD-MM-YYYY>.csv.   Press 'q' to quit.
"""

import argparse
import csv
import os
import platform
from datetime import datetime

import cv2

from face_engine import DEFAULT_THRESHOLD, FaceEngine, Gallery

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")
BACKGROUND_PATH = os.path.join(BASE_DIR, "background.png")
COL_NAMES = ["NAME", "TIME", "SCORE"]

GREEN, RED = (60, 180, 60), (50, 50, 255)


def speak(text):
    """Voice feedback on Windows (needs pywin32); otherwise just prints."""
    if platform.system() == "Windows":
        try:
            from win32com.client import Dispatch
            Dispatch("SAPI.SpVoice").Speak(text)
            return
        except Exception:
            pass
    print(f"[voice] {text}")


def attendance_file(now):
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    return os.path.join(ATTENDANCE_DIR, f"Attendance_{now.strftime('%d-%m-%Y')}.csv")


def already_marked_today(path):
    if not os.path.isfile(path):
        return set()
    with open(path, newline="") as f:
        return {row[0] for row in csv.reader(f) if row and row[0] != COL_NAMES[0]}


def mark(path, name, score, now):
    new_file = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(COL_NAMES)
        writer.writerow([name, now.strftime("%H:%M:%S"), f"{score:.2f}"])


def draw_face(frame, face, label, colour):
    x1, y1, x2, y2 = (int(v) for v in face.box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    cv2.rectangle(frame, (x1, max(y1 - 32, 0)), (x2, y1), colour, -1)
    cv2.putText(frame, label, (x1 + 4, max(y1 - 9, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for px, py in face.landmarks:
        cv2.circle(frame, (int(px), int(py)), 2, (0, 255, 255), -1)


def main():
    parser = argparse.ArgumentParser(description="Face-recognition attendance")
    parser.add_argument("--source", default="0", help="camera index or video file (default 0)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="minimum cosine similarity to accept an identity")
    parser.add_argument("--confirm", type=int, default=5,
                        help="consecutive frames a person must be recognised before attendance is marked")
    parser.add_argument("--min_face", type=int, default=60, help="ignore faces narrower than this many pixels")
    parser.add_argument("--det_size", type=int, default=640, help="YOLO input size (320 is faster)")
    parser.add_argument("--no-display", action="store_true", help="run without a window")
    parser.add_argument("--no-background", action="store_true", help="plain video instead of the background image")
    parser.add_argument("--max_frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    args = parser.parse_args()

    gallery = Gallery.load()
    if not gallery.names:
        raise SystemExit("Nobody is registered yet. Run:  python register.py --name YourName")
    print(f"Registered people: {gallery.people}")

    engine = FaceEngine(det_size=args.det_size)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source {args.source!r}. Check the camera permission/connection.")

    background = None
    if not args.no_background and not args.no_display and os.path.isfile(BACKGROUND_PATH):
        background = cv2.imread(BACKGROUND_PATH)
        if background is not None:   # the image still says "press O" - attendance is automatic now
            h, w = background.shape[:2]
            cv2.rectangle(background, (int(w * 0.04), int(h * 0.915)), (int(w * 0.90), int(h * 0.99)),
                          (255, 255, 255), -1)
            cv2.putText(background, "ATTENDANCE IS MARKED AUTOMATICALLY  -  PRESS Q TO QUIT",
                        (int(w * 0.05), int(h * 0.958)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)

    streak = {}                       # name -> consecutive frames recognised
    marked = already_marked_today(attendance_file(datetime.now()))
    frames = 0
    print("Running. Press 'q' to quit." if not args.no_display else "Running (no window).")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        seen = set()
        for face in engine.analyse(frame, min_face_px=args.min_face):
            name, score = gallery.identify(face.embedding, args.threshold)
            if name is None:
                draw_face(frame, face, f"Unknown {score:.2f}", RED)
                continue
            seen.add(name)
            streak[name] = streak.get(name, 0) + 1
            label = f"{name} {score:.2f}" + ("  present" if name in marked else "")
            draw_face(frame, face, label, GREEN)

            if streak[name] >= args.confirm and name not in marked:
                now = datetime.now()
                mark(attendance_file(now), name, score, now)
                marked.add(name)
                print(f"Marked present: {name} at {now.strftime('%H:%M:%S')} (score {score:.2f})")
                speak(f"Attendance taken for {name}")
        for name in list(streak):          # anyone not seen this frame loses their streak
            if name not in seen:
                streak[name] = 0

        if not args.no_display:
            cv2.putText(frame, f"Present today: {len(marked)}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            display = frame
            if background is not None:
                display = background.copy()
                display[162:162 + 480, 55:55 + 640] = cv2.resize(frame, (640, 480))
            cv2.imshow("Attendance", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.max_frames and frames >= args.max_frames:
            break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print(f"Done. {frames} frames processed. Marked present today: {sorted(marked) or 'nobody'}")


if __name__ == "__main__":
    main()
