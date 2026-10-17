"""
register.py - add a person to the gallery.

From the webcam (look at the camera and slowly turn your head a little):
    python register.py --name Safina

From photos instead (one sub-folder per person: enroll/Safina/1.jpg, enroll/Safina/2.jpg ...):
    python register.py --from_folder enroll

Each accepted frame/photo becomes one 512-number embedding stored in data/gallery.npz.
Registering the same name again replaces that person's old samples.
"""

import argparse
import os
import time

import cv2

from face_engine import FaceEngine, Gallery, GALLERY_PATH

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_source(value):
    return int(value) if str(value).isdigit() else value


def register_from_webcam(engine, name, source, samples, gap, min_face, min_conf, show):
    cap = cv2.VideoCapture(parse_source(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source {source!r}. Check the camera permission/connection.")

    embeddings, last_saved = [], 0.0
    print(f"Collecting {samples} samples for '{name}'. Look at the camera and turn your head slightly. "
          "Press 'q' to stop.")
    while len(embeddings) < samples:
        ok, frame = cap.read()
        if not ok:
            print("Video source ended / frame could not be read.")
            break

        faces = engine.analyse(frame, min_face_px=min_face)
        status = f"{len(embeddings)}/{samples}"
        if len(faces) > 1:
            status += "  - only one person in the frame, please"
        elif len(faces) == 1 and faces[0].conf >= min_conf and time.time() - last_saved >= gap:
            embeddings.append(faces[0].embedding)
            last_saved = time.time()

        if show:
            for f in faces:
                x1, y1, x2, y2 = (int(v) for v in f.box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("Register", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if show:
        cv2.destroyAllWindows()
    return embeddings


def register_from_folder(engine, folder, gallery):
    added = 0
    for person in sorted(os.listdir(folder)):
        person_dir = os.path.join(folder, person)
        if not os.path.isdir(person_dir):
            continue
        embeddings = []
        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            img = cv2.imread(os.path.join(person_dir, fname))
            faces = engine.analyse(img) if img is not None else []
            if len(faces) != 1:
                print(f"  skipped {person}/{fname}: found {len(faces)} faces (need exactly 1)")
                continue
            embeddings.append(faces[0].embedding)
        if embeddings:
            gallery.add(person, embeddings)
            added += 1
            print(f"Registered {person}: {len(embeddings)} photos")
        else:
            print(f"No usable photos for {person}")
    return added


def main():
    parser = argparse.ArgumentParser(description="Register people for the attendance system")
    parser.add_argument("--name", help="person's name (webcam registration)")
    parser.add_argument("--from_folder", help="folder with one sub-folder of photos per person")
    parser.add_argument("--source", default="0", help="camera index or a video file (default 0)")
    parser.add_argument("--samples", type=int, default=20, help="samples to collect from the webcam")
    parser.add_argument("--gap", type=float, default=0.3, help="seconds between saved samples")
    parser.add_argument("--min_face", type=int, default=80, help="smallest face width accepted, in pixels")
    parser.add_argument("--min_conf", type=float, default=0.6, help="smallest detector confidence accepted")
    parser.add_argument("--no-display", action="store_true", help="do not open a window")
    args = parser.parse_args()

    engine = FaceEngine()
    gallery = Gallery.load()

    if args.from_folder:
        if register_from_folder(engine, args.from_folder, gallery):
            gallery.save()
            print(f"Saved. People now registered: {gallery.people}")
        else:
            print("Nothing registered.")
        return

    name = (args.name or input("Enter your name: ")).strip()
    if not name:
        raise ValueError("Name cannot be empty.")
    embeddings = register_from_webcam(engine, name, args.source, args.samples, args.gap,
                                      args.min_face, args.min_conf, show=not args.no_display)
    if len(embeddings) < 5:
        print(f"Only {len(embeddings)} good samples collected - NOT saved. "
              "Try again with better light and face the camera.")
        return
    gallery.add(name, embeddings)
    gallery.save()
    print(f"Saved {len(embeddings)} samples for '{name}'. People now registered: {gallery.people}")


if __name__ == "__main__":
    main()
