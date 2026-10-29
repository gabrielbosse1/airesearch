"""Synthetic multimodal dataset (raw bytes only).

Samples:
  * text   : short phrases -> utf-8 bytes
  * image  : random geometric shapes -> grayscale pixels (bytes)
  * audio  : simple tones / noise -> raw 8-bit PCM bytes

Every sample carries a ``concept`` label so the concept bank can be
populated with grouped examples. All data is raw byte arrays (ints 0-255).
"""

import json
import os
import random

import numpy as np


SHAPES = ["rect", "line", "circle", "noise"]
WORDS = [
    "cat", "dog", "red", "blue", "sun", "moon", "star", "tree", "car", "bird",
    "fish", "fire", "ice", "snow", "rain", "wind", "sand", "rock", "gold",
    "silk", "wolf", "fox", "bee", "ant", "cow", "pig", "hen", "duck",
]
PHRASES = ["{a} {b}", "{a} and {b}", "a {a}", "the {a}", "{a} {a}", "{b} {a}"]


def _rand_text(rng):
    template = rng.choice(PHRASES)
    a = rng.choice(WORDS)
    b = rng.choice([w for w in WORDS if w != a])
    text = template.format(a=a, b=b)
    return list(text.encode("utf-8")), text


def _rand_image(rng):
    size = 8
    canvas = np.zeros((size, size), dtype=np.uint8)
    n_objs = rng.randint(1, 4)
    for _ in range(n_objs):
        shape = rng.choice(SHAPES)
        color = rng.randint(60, 255)
        if shape == "rect":
            x0, y0 = rng.randint(0, size), rng.randint(0, size)
            x1, y1 = rng.randint(0, size), rng.randint(0, size)
            x0, x1 = sorted([x0, x1]); y0, y1 = sorted([y0, y1])
            canvas[y0:y1, x0:x1] = color
        elif shape == "line":
            x0, y0 = rng.randint(0, size), rng.randint(0, size)
            x1, y1 = rng.randint(0, size), rng.randint(0, size)
            # Bresenham-ish: draw a few points between
            for t in np.linspace(0, 1, num=size):
                x = int(round(x0 + (x1 - x0) * t))
                y = int(round(y0 + (y1 - y0) * t))
                if 0 <= x < size and 0 <= y < size:
                    canvas[y, x] = color
        elif shape == "circle":
            cx, cy = rng.randint(0, size), rng.randint(0, size)
            r = rng.randint(1, 3)
            yy, xx = np.ogrid[:size, :size]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            canvas[mask] = color
        else:  # noise
            canvas = np.array(
                [[rng.randint(0, 255) for _ in range(size)] for _ in range(size)],
                dtype=np.uint8)
    return canvas.reshape(-1).tolist(), canvas


def _rand_audio(rng):
    n = 64
    kind = rng.choice(["tone", "noise", "click", "chirp"])
    t = np.arange(n) / 64.0
    if kind == "tone":
        f = rng.uniform(2, 20)
        sig = np.sin(2 * np.pi * f * t)
    elif kind == "noise":
        sig = np.array([rng.gauss(0, 1) for _ in range(n)])
    elif kind == "click":
        sig = np.zeros(n)
        sig[rng.randint(0, n - 1)] = 1.0
    else:  # chirp
        f0, f1 = rng.uniform(2, 8), rng.uniform(8, 20)
        sig = np.sin(2 * np.pi * np.linspace(f0, f1, n) * t)
    # Normalize to 8-bit unsigned PCM bytes.
    sig = sig / (np.max(np.abs(sig)) + 1e-6)
    pcm = ((sig + 1.0) / 2.0 * 255).astype(np.uint8)
    return pcm.tolist(), kind


def generate_sample(rng):
    modality = rng.choice(["text", "image", "audio"])
    if modality == "text":
        bytes_, text = _rand_text(rng)
        concept = text.split()[0] if text.split() else "word"
    elif modality == "image":
        bytes_, canvas = _rand_image(rng)
        concept = "img:" + "+".join(sorted(set(SHAPES)))
        # give each image a simpler dominant concept
        concept = "shape"
    else:
        bytes_, kind = _rand_audio(rng)
        concept = "audio:" + kind
    return {
        "modality": modality,
        "concept": concept,
        "bytes": bytes_,
    }


def generate_dataset(n_samples, seed=42):
    rng = random.Random(seed)
    return [generate_sample(rng) for _ in range(n_samples)]


def corrupt_sequence(seq, f, mode, rng):
    """Corrupt fraction f of a byte list; return new list (same length)."""
    b = list(seq)
    n = len(b)
    if n == 0 or f <= 0:
        return b
    k = max(1, int(round(f * n)))
    if mode == "substitute":
        idx = rng.sample(range(n), min(k, n))
        for i in idx:
            b[i] = rng.randint(0, 255)
    elif mode == "insert":
        out = []
        inserts_left = k
        for i, byte in enumerate(b):
            out.append(byte)
            n_ins = inserts_left // max(n - i, 1)
            inserts_left -= n_ins
            for _ in range(n_ins):
                out.append(rng.randint(0, 255))
        b = out[: len(seq)] if len(seq) else out
    elif mode == "delete":
        kill = set(rng.sample(range(n), min(k, n)))
        b = [byte for i, byte in enumerate(b) if i not in kill]
        b = b + [0] * (n - len(b))  # pad back to original length
    return b


def corrupt_bytes(b, f, mode, rng):
    """Corrupt fraction f of a byte list; return new list."""
    return corrupt_sequence(list(b), f, mode, rng)


def noisy_samples(samples, f, mode, seed=1234):
    rng = random.Random(seed)
    out = []
    for s in samples:
        cb = corrupt_sequence(s["bytes"], f, mode, rng)
        out.append({"modality": s["modality"], "concept": s["concept"],
                    "bytes": cb})
    return out


def save_jsonl(samples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def main():
    import yaml

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    d = cfg["data"]
    train = generate_dataset(d["train_samples"], seed=42)
    val = generate_dataset(d["val_samples"], seed=123)
    test = generate_dataset(d["test_samples"], seed=456)

    save_jsonl(train, os.path.join(d["processed_dir"], "train.jsonl"))
    save_jsonl(val, os.path.join(d["processed_dir"], "val.jsonl"))
    save_jsonl(test, os.path.join(d["processed_dir"], "test.jsonl"))

    from collections import Counter
    print("Dataset generated:")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        c = Counter(s["modality"] for s in split)
        c2 = Counter(s["concept"] for s in split)
        print(f"  {name}: {len(split)} samples  modality={dict(c)}")
        print(f"            concept sample={dict(list(c2.items())[:5])}")
    print(f"Saved to {d['processed_dir']}/")


if __name__ == "__main__":
    main()
