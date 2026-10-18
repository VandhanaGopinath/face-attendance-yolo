"""
lfw_benchmark.py - test the recognition pipeline on LFW (Labeled Faces in the Wild), the standard
public face-verification benchmark.

    pip install scikit-learn          # only needed here, to download the dataset (~200 MB, once)
    python lfw_benchmark.py

The task ("face verification"): for each pair of photos, decide whether they show the SAME person.
LFW ships official lists of such pairs; this script uses them exactly as published:

  * 2,200 "development train" pairs  -> used ONLY to choose the similarity threshold
  * 1,000 "development test"  pairs  -> used ONLY to report accuracy (never used to tune anything)

For every photo the pipeline is the same as in the attendance app: YOLO face detection (the face
closest to the image centre is the target), landmark alignment, ArcFace embedding. A pair is
called "same person" when the cosine similarity of the two embeddings is >= threshold.

If YOLO finds no face in a photo, the central part of the photo is used instead (LFW photos are
centred on the face) and the photo is counted under "fallback", so you can see how often it happened.

Takes roughly 10 minutes on a laptop CPU for the full run. For a quick check: --max_pairs 200
"""

import argparse
import os
import time

import cv2
import numpy as np

from face_engine import DEFAULT_THRESHOLD, FaceEngine

PAIR_FILES = {"train": "pairsDevTrain.txt", "test": "pairsDevTest.txt"}


def locate_lfw(data_home):
    """Make sure LFW is on disk (scikit-learn downloads it) and return the lfw_home folder."""
    try:
        from sklearn.datasets import fetch_lfw_pairs, get_data_home
    except ImportError:
        raise SystemExit("scikit-learn is needed to download LFW:  pip install scikit-learn")
    try:
        # Asking for 2x2-pixel crops makes scikit-learn download + unpack LFW without
        # loading thousands of full images into memory.
        for subset in ("train", "test"):
            fetch_lfw_pairs(subset=subset, data_home=data_home, slice_=(slice(0, 2), slice(0, 2)),
                            resize=1.0, color=False)
    except Exception as exc:
        raise SystemExit(
            f"Could not get the LFW dataset: {exc}\n\n"
            "If you are behind a firewall, download these from http://vis-www.cs.umass.edu/lfw/ :\n"
            "  lfw-funneled.tgz, pairsDevTrain.txt, pairsDevTest.txt, pairs.txt\n"
            "and place them in  <home>/scikit_learn_data/lfw_home/  (unpack the .tgz there).")
    return os.path.join(get_data_home(data_home=data_home), "lfw_home")


def read_pairs(lfw_home, subset):
    """-> list of (path_a, path_b, same_person 0/1) from the official pairs file."""
    folder = os.path.join(lfw_home, "lfw_funneled")
    listing = {}

    def image(name, idx):
        if name not in listing:
            listing[name] = sorted(os.listdir(os.path.join(folder, name)))
        return os.path.join(folder, name, listing[name][idx - 1])

    pairs = []
    with open(os.path.join(lfw_home, PAIR_FILES[subset]), encoding="utf-8") as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) == 3:
                pairs.append((image(p[0], int(p[1])), image(p[0], int(p[2])), 1))
            elif len(p) == 4:
                pairs.append((image(p[0], int(p[1])), image(p[2], int(p[3])), 0))
    return pairs


def balanced_subset(pairs, n):
    """First n//2 same-person pairs and n//2 different-person pairs (the files list all 'same' first)."""
    if not n:
        return pairs
    same = [p for p in pairs if p[2] == 1][: n // 2]
    diff = [p for p in pairs if p[2] == 0][: n // 2]
    return same + diff


def embed_photo(engine, path):
    """-> (embedding, used_fallback). Target face = detection closest to the image centre."""
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"could not read {path}")
    h, w = img.shape[:2]
    faces = engine.analyse(img)
    if faces:
        cx, cy = w / 2, h / 2
        face = min(faces, key=lambda f: ((f.box[0] + f.box[2]) / 2 - cx) ** 2 + ((f.box[1] + f.box[3]) / 2 - cy) ** 2)
        return face.embedding, False
    side = int(min(h, w) * 0.5)                                    # fallback: central crop
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = cv2.resize(img[y0:y0 + side, x0:x0 + side], (112, 112))
    return engine.embed_aligned([crop])[0], True


def score_pairs(engine, pairs, label):
    cache, fallbacks, t0 = {}, 0, time.time()
    scores, truth = [], []
    paths = sorted({p for a, b, _ in pairs for p in (a, b)})
    for i, path in enumerate(paths, 1):
        cache[path], used = embed_photo(engine, path)
        fallbacks += used
        if i % 250 == 0 or i == len(paths):
            print(f"  [{label}] {i}/{len(paths)} photos embedded ({time.time() - t0:.0f}s)")
    for a, b, same in pairs:
        scores.append(float(cache[a] @ cache[b]))
        truth.append(same)
    return np.array(scores), np.array(truth), len(paths), fallbacks


def best_threshold(scores, truth):
    candidates = np.arange(-0.2, 1.0, 0.005)
    acc = np.array([np.mean((scores >= t) == truth.astype(bool)) for t in candidates])
    best = np.flatnonzero(acc == acc.max())
    return float(candidates[best[len(best) // 2]]), float(acc.max())


def auc(scores, truth):
    """Probability that a random same-person pair scores higher than a random different-person pair."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = int(truth.sum()), int((1 - truth).sum())
    return float((ranks[truth == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    parser = argparse.ArgumentParser(description="LFW face-verification benchmark")
    parser.add_argument("--data_home", default=None, help="where scikit-learn stores datasets (default ~/scikit_learn_data)")
    parser.add_argument("--max_pairs", type=int, default=0, help="use only N pairs per split (balanced); 0 = all")
    parser.add_argument("--skip_train", action="store_true", help="do not tune: use --threshold as is")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--out", default="lfw_results.md")
    args = parser.parse_args()

    lfw_home = locate_lfw(args.data_home)
    engine = FaceEngine()

    test_pairs = balanced_subset(read_pairs(lfw_home, "test"), args.max_pairs)
    print(f"Test split: {len(test_pairs)} pairs")
    test_scores, test_truth, n_test_imgs, test_fb = score_pairs(engine, test_pairs, "test")

    if args.skip_train:
        threshold, train_acc, n_train, n_train_imgs = args.threshold, None, 0, 0
    else:
        train_pairs = balanced_subset(read_pairs(lfw_home, "train"), args.max_pairs)
        print(f"Train split (threshold selection only): {len(train_pairs)} pairs")
        train_scores, train_truth, n_train_imgs, _ = score_pairs(engine, train_pairs, "train")
        threshold, train_acc = best_threshold(train_scores, train_truth)
        n_train = len(train_pairs)

    pred = test_scores >= threshold
    acc = float(np.mean(pred == test_truth.astype(bool)))
    acc_default = float(np.mean((test_scores >= DEFAULT_THRESHOLD) == test_truth.astype(bool)))
    tar = float(np.mean(pred[test_truth == 1]))        # same-person pairs correctly accepted
    far = float(np.mean(pred[test_truth == 0]))        # different-person pairs wrongly accepted

    lines = [
        "# LFW face-verification results",
        "",
        f"- Test pairs: {len(test_pairs)} ({int(test_truth.sum())} same-person, {int((1 - test_truth).sum())} different-person), {n_test_imgs} photos",
        f"- Threshold: {threshold:.3f} " + (f"(chosen on {n_train} train pairs, train accuracy {train_acc * 100:.2f}%)" if train_acc is not None else "(fixed, not tuned)"),
        f"- **Verification accuracy on the test pairs: {acc * 100:.2f}%**",
        f"- Same-person pairs accepted: {tar * 100:.2f}%   |   different-person pairs wrongly accepted: {far * 100:.2f}%",
        f"- ROC-AUC: {auc(test_scores, test_truth):.4f}",
        f"- Accuracy at the app's default threshold ({DEFAULT_THRESHOLD}): {acc_default * 100:.2f}%",
        f"- Photos where YOLO found no face and the central-crop fallback was used: {test_fb} of {n_test_imgs}",
        "",
        "Protocol: official LFW dev-train pairs choose the threshold, official dev-test pairs measure accuracy.",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
