"""Export and parity-check the selected Task 4 model for the course browser app."""
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms
import task4_experiments as exp

ROOT = Path(__file__).resolve().parent
CHECKPOINT = exp.RESULTS / "checkpoints" / "combo_frozen_color_adamw_smoothing_group_historical_seed2_best.pt"
INFO = {"classes": ["apple", "cherry", "chestnut", "maple"], "input_size": 96,
        "mean": exp.MEAN, "std": exp.STD}


def main():
    model = exp.create_model("mobilenet", frozen=True)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    model.eval().cpu()
    onnx_path = ROOT / "leaf_model.onnx"
    original = ROOT / "task4_results" / "leaf_model_task2_original.onnx"
    if not original.exists():
        shutil.copy2(onnx_path, original)
    torch.onnx.export(model, torch.zeros(1, 3, 96, 96), str(onnx_path),
                      input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamo=False, external_data=False)
    exported = onnx.load(str(onnx_path))
    metadata = exported.metadata_props.add()
    metadata.key, metadata.value = "model_info", json.dumps(INFO)
    onnx.save(exported, str(onnx_path))
    (ROOT / "model_info.json").write_text(json.dumps(INFO, indent=2) + "\n", encoding="utf-8")

    transform = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor(),
                                    transforms.Normalize(exp.MEAN, exp.STD)])
    image = Image.open(ROOT / "check_image.jpg").convert("RGB")
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        expected = model(x).softmax(1).numpy()[0]
    (ROOT / "check_expected.json").write_text(
        json.dumps(dict(zip(INFO["classes"], map(float, expected))), indent=2) + "\n", encoding="utf-8")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    observed = session.run(["logits"], {"image": x.numpy()})[0]
    error = float(np.max(np.abs(observed - model(x).detach().numpy())))
    if error > 1e-4:
        raise RuntimeError(f"ONNX parity failed: maximum logit error {error}")
    print(json.dumps({"onnx": str(onnx_path), "checkpoint": str(CHECKPOINT),
                      "size_bytes": onnx_path.stat().st_size, "max_abs_logit_error": error,
                      "check_probabilities": dict(zip(INFO["classes"], map(float, expected)))}, indent=2))


if __name__ == "__main__":
    main()
