"""Evaluate the supplied phone export on the historical validation set.

Usage: python evaluate_historical.py --data C:/path/to/leaves
Requires onnxruntime, Pillow and numpy.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import onnxruntime as ort


def tensor(image, size, mean, std):
    image = image.resize((size, size), Image.Resampling.BILINEAR).convert("RGB")
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return ((pixels - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)).transpose(2, 0, 1)[None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Historical dataset root containing train/ and val/")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("leaf_model.onnx"))
    parser.add_argument("--info", type=Path, default=Path(__file__).with_name("model_info.json"))
    args = parser.parse_args()
    info = json.loads(args.info.read_text(encoding="utf-8"))
    classes, size = info["classes"], info["input_size"]
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name, output_name = session.get_inputs()[0].name, session.get_outputs()[0].name
    rng = random.Random(20261112)
    correct = {"regular": 0, "camera_like": 0}
    total = 0
    for label, name in enumerate(classes):
        for path in sorted((args.data / "val" / name).glob("*")):
            with Image.open(path) as src:
                image = src.convert("RGB")
            base = image.resize((size, size), Image.Resampling.BILINEAR)
            crop_fraction = rng.uniform(0.72, 0.96)
            side = int(size * crop_fraction)
            left, top = rng.randint(0, size - side), rng.randint(0, size - side)
            camera = base.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.BILINEAR)
            camera = camera.rotate(rng.uniform(-20, 20), resample=Image.Resampling.BILINEAR)
            camera = ImageEnhance.Color(camera).enhance(rng.uniform(0.65, 1.35))
            camera = ImageEnhance.Brightness(camera).enhance(rng.uniform(0.8, 1.2))
            for key, sample in (("regular", base), ("camera_like", camera)):
                scores = session.run([output_name], {input_name: tensor(sample, size, info["mean"], info["std"])})[0].reshape(-1)
                if int(scores.argmax()) == label:
                    correct[key] += 1
            total += 1
    print(json.dumps({"model": str(args.model), "data": str(args.data), "seed": 20261112,
                      "samples": total, "accuracy": {k: round(v / total, 4) for k, v in correct.items()}}, indent=2))


if __name__ == "__main__":
    main()
