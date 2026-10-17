"""
evaluate.py - measure how well recognition works, on photos YOU provide.

Put photos in one sub-folder per person (at least 3 people, and 5+ photos each is better,
taken on different days / in different light - ideally from the webcam you will really use):

    data/eval/Safina/*.jpg
    data/eval/Friend1/*.jpg
    data/eval/Friend2/*.jpg

    python evaluate.py --data data/eval

What it does (repeated --runs times with random splits, then averaged):
  * a few photos per person are "registered" (--enroll); the rest are test photos
  * some people are held out completely (never registered) to play the role of strangers
  * every test photo is compared against the registered people

Metrics reported at your threshold
  * Identification accuracy - registered people recognised as the RIGHT person
  * False reject rate       - registered people wrongly called Unknown or someone else
  * False accept rate       - strangers wrongly accepted as a registered person
and the threshold where false-accept and false-reject are equal (the "equal error rate"),
plus a small threshold table so you can pick the one you want.
"""

import argparse
import os

import cv2
import numpy as np

from face_engine import DEFAULT_THRESHOLD, FaceEngine

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_embeddings(engine, root):
    """-> {person: [embedding, ...]} using the largest face in each image."""
    data, no_face = {}, 0
    for person in sorted(os.listdir(root)):
        pdir = os.path.join(root, person)
        if not os.path.isdir(pdir):
            continue
        for fname in sorted(os.listdir(pdir)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            img = cv2.imread(os.path.join(pdir, fname))
            faces = engine.analyse(img) if img is not None else []
            if not faces:
                no_face += 1
                print(f"  no face found in {person}/{fname}")
                continue
            data.setdefault(person, []).append(faces[0].embedding)
    return {p: np.vstack(v) for p, v in data.items()}, no_face


def one_run(data, enroll, n_unknown, rng, thresholds):
    people = [p for p in data if len(data[p]) > enroll]          # need photos left over for testing
    if len(people) < 3:
        return None
    unknown = set(rng.choice(people, size=min(n_unknown, len(people) - 2), replace=False)) if n_unknown else set()
    registered = [p for p in people if p not in unknown]

    centroids = []
    test_known, test_unknown = [], []                  # (true person, embedding)
    for p in registered:
        order = rng.permutation(len(data[p]))
        enrol_idx, test_idx = order[:enroll], order[enroll:]
        c = data[p][enrol_idx].mean(axis=0)
        centroids.append(c / np.linalg.norm(c))
        test_known += [(p, data[p][i]) for i in test_idx]
    for p in unknown:
        test_unknown += [(p, e) for e in data[p]]
    centroids = np.vstack(centroids)

    def best(emb):
        s = centroids @ emb
        j = int(np.argmax(s))
        return registered[j], float(s[j])

    known = [(p, *best(e)) for p, e in test_known]       # (true, predicted, score)
    unk = [best(e)[1] for _, e in test_unknown]          # best score a stranger achieved
    result = {}
    for t in thresholds:
        correct = sum(1 for p, pred, s in known if s >= t and pred == p)
        result[t] = (correct / max(len(known), 1),
                     (sum(1 for s in unk if s >= t) / len(unk)) if unk else float("nan"))
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate face recognition on your own photos")
    parser.add_argument("--data", default="data/eval", help="folder with one sub-folder of photos per person")
    parser.add_argument("--enroll", type=int, default=3, help="photos per person used for registration")
    parser.add_argument("--unknown", type=int, default=1, help="people held out as strangers in each run")
    parser.add_argument("--runs", type=int, default=50, help="random splits to average")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        raise SystemExit(f"Folder not found: {args.data}")
    engine = FaceEngine()
    data, no_face = load_embeddings(engine, args.data)
    total = sum(len(v) for v in data.values())
    print(f"\n{len(data)} people, {total} usable photos ({no_face} photos had no detectable face)")
    for p, v in data.items():
        print(f"  {p}: {len(v)} photos")

    thresholds = np.round(np.arange(0.20, 0.801, 0.05), 2).tolist()
    if args.threshold not in thresholds:
        thresholds = sorted(thresholds + [args.threshold])
    rng = np.random.default_rng(args.seed)
    runs = [r for r in (one_run(data, args.enroll, args.unknown, rng, thresholds) for _ in range(args.runs)) if r]
    if not runs:
        raise SystemExit("Need at least 3 people with more than --enroll photos each.")

    acc = {t: np.mean([r[t][0] for r in runs]) for t in thresholds}
    far = {t: np.nanmean([r[t][1] for r in runs]) if args.unknown else float("nan") for t in thresholds}

    print(f"\nAveraged over {len(runs)} random splits ({args.enroll} registration photos per person)\n")
    print(f"{'threshold':>9} | {'correct ID rate':>15} | {'stranger accepted':>17}")
    print("-" * 49)
    for t in thresholds:
        mark = "  <-- current" if t == args.threshold else ""
        far_txt = f"{far[t] * 100:15.1f} %" if not np.isnan(far[t]) else "             n/a"
        print(f"{t:9.2f} | {acc[t] * 100:13.1f} % | {far_txt}{mark}")

    if args.unknown:
        # equal error rate: where wrongly-rejected registered people == wrongly-accepted strangers
        gap = {t: abs((1 - acc[t]) - far[t]) for t in thresholds}
        eer_t = min(gap, key=gap.get)
        print(f"\nRoughly equal error at threshold {eer_t:.2f}: "
              f"~{(1 - acc[eer_t]) * 100:.1f} % of registered people rejected / "
              f"{far[eer_t] * 100:.1f} % of strangers accepted.")
    print("\n'Correct ID rate' counts a test photo as correct only if it is accepted AND matched to the right person.")


if __name__ == "__main__":
    main()
