# Task 4 — Leaf classifier experiments

All rows below are mean accuracy ± standard deviation across seeds 0, 1, and 2. Group validation contains 62 images; historical validation contains 120. The same fixed, camera-like transformation of group validation images was used for each variant. Checkpoints were selected by regular group validation accuracy.

| Variant | Group regular | Group camera-like | Historical regular |
|---|---:|---:|---:|
| Task 3 baseline | 76.9 ± 6.7% | 62.9 ± 2.8% | 45.6 ± 8.2% |
| ColorJitter only | 74.2 ± 0.0% | 64.0 ± 4.1% | — |
| Dropout only | 73.7 ± 2.5% | 57.0 ± 1.9% | — |
| Global average pooling only | 68.8 ± 1.9% | 55.9 ± 0.9% | — |
| AdamW, weight decay 1e-4 only | 75.8 ± 6.5% | 60.8 ± 4.1% | — |
| Label smoothing 0.1 only | 78.5 ± 4.9% | 61.3 ± 3.2% | — |
| Pretrained MobileNet, frozen features | **89.2 ± 0.9%** | 80.6 ± 3.2% | — |
| Pretrained MobileNet, fully fine-tuned | 83.3 ± 6.5% | 78.5 ± 2.5% | — |
| MobileNet, augmentation + cosine schedule, 30 epochs | 85.5 ± 2.8% | 81.7 ± 3.4% | — |
| CNN + ColorJitter + AdamW + label smoothing | 76.9 ± 0.9% | 65.1 ± 4.9% | 43.9 ± 7.5% |
| MobileNet + ColorJitter + AdamW + smoothing + cosine | 84.9 ± 2.5% | 80.6 ± 1.6% | 54.4 ± 2.4% |
| Frozen MobileNet + ColorJitter + cosine, 30 epochs | 88.7 ± 0.0% | 83.3 ± 0.9% | 47.5 ± 2.9% |
| Frozen MobileNet + ColorJitter + AdamW + smoothing | 88.2 ± 0.9% | **86.0 ± 0.9%** | 49.2 ± 2.2% |
| Task 3 baseline, group + historical training | 77.4 ± 1.6% | 61.3 ± 9.7% | 81.7 ± 4.3% |
| Frozen MobileNet + ColorJitter + AdamW + smoothing, group + historical training | 84.9 ± 0.9% | 82.8 ± 3.4% | **84.2 ± 2.5%** |

## What the changes showed

- **Pretraining and freezing helped most:** frozen MobileNet improved over the Task 3 baseline by 12.3 percentage points on regular group validation and 17.7 points on camera-like validation. These gains are larger than the observed seed variation, though the small validation set limits certainty.
- **Augmentation and training schedule:** ColorJitter alone raised camera-like accuracy by only 1.1 points, within the noise. The 30-epoch MobileNet with augmentation and cosine scheduling reached 81.7% camera-like accuracy; compared with ordinary full fine-tuning, the 3.2-point gain is modest relative to run variation. Frozen MobileNet combinations with ColorJitter reached 83.3–86.0% camera-like accuracy.
- **Historical training photos:** adding them to the recommended combination raised historical validation accuracy from 49.2% to 84.2%, a large gain beyond the observed seed variation. Group accuracy fell from 88.2% to 84.9%, while camera-like accuracy fell from 86.0% to 82.8%. This is the better tradeoff if the phone should handle photos from other students and phones.
- **What did not help:** dropout and global average pooling reduced both group scores. AdamW or label smoothing alone did not produce a clear improvement over baseline. Full fine-tuning was less consistent and scored below frozen features. Small differences of a few points are not decisive: one group validation image changes accuracy by 1.6 points, and the baseline regular score varied by 6.7 points across seeds.

## Phone recommendation

I would put the **frozen pretrained MobileNetV3 Small with ColorJitter, AdamW (weight decay 1e-4), and label smoothing 0.1, trained on group plus historical training images** on the phone. It achieved 84.9% regular group, 82.8% camera-like group, and 84.2% historical validation accuracy across three seeds. The exported `leaf_model.onnx` uses seed 2, best checkpoint epoch 5; its logits matched PyTorch within 6.2×10⁻⁶ on the check image. The course camera app/device check remains to be done; it was unavailable during this work.
