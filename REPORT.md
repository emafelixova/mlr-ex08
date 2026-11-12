# Task 4 — improving the leaf classifier

## Result and recommendation

The recommended phone model is the **frozen pretrained MobileNetV3 Small with ColorJitter, AdamW (weight decay 1e-4), and label smoothing 0.1, trained on both the group and historical training images**. Across seeds 0–2, its best group-validation checkpoints averaged **84.9% regular**, **82.8% camera-like**, and **84.2% on the historical validation photos**. This trades about 3 percentage points of group accuracy versus the same recipe trained on group images alone (88.2% / 86.0%) for about 35 percentage points better accuracy on the historical validation photos (49.2% → 84.2%). Since the intended app should handle leaves photographed by other students and phones, that generalization is the more useful tradeoff. For a deployment restricted to the exact group photo domain, the group-only version scored higher on both group sets.

The exported `leaf_model.onnx` uses seed 2, selected best checkpoint epoch 5. ONNX Runtime and PyTorch logits agreed within **6.2×10⁻⁶ maximum absolute error** on the required check image. The ONNX export reached **84.17% (101/120)** regular accuracy and **87.50% (105/120)** on a supplementary camera-like transformation of the historical validation images. The app/device check was not possible in this session: no course camera app or connected phone was available, and the browser controller could not verify its active page. The model and matching `model_info.json`, `check_image.jpg`, and `check_expected.json` are ready for that check.

## Data and fair comparison

The linked group archive was retrieved and used in the notebook's existing split. It contains 244 training images (apple 68, cherry 75, chestnut 41, maple 60) and 62 regular validation images (17, 19, 11, 15). No group validation image was used for training. The historical archive contributes 700 training images and 120 validation images (175/30 per species); its validation images were never included in any training set.

The fixed baseline is the final Task 3 pipeline from the notebook: `LeafNetBN` with four 3×3 conv/ReLU/batch-normalization/max-pool blocks (16, 32, 64, 64 channels), flatten, 64-unit hidden layer, and four logits. Input is 96×96 RGB, normalized with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`; batch size 32; Adam at 0.001; 15 epochs; seed 0, 1, or 2. Training augmentation is the notebook recipe: `RandomResizedCrop(96, scale=(0.6, 1.0))`, horizontal and vertical flips, and `RandomRotation(20)`.

The camera-like group validation images were generated once from the 62 regular validation images with seed 240517, then reused for every run: random crop fraction 0.68–0.95, resize to 96×96, random rotation ±20°, color multiplier 0.65–1.35, and brightness multiplier 0.8–1.2. This is a repeatable stress set, not 62 new independent leaves.

All group-only rows use the same group training split; the two mixed-data rows add only historical training photos. Each variant saves both its best checkpoint (selected by regular group validation accuracy) and final-epoch checkpoint. Results in the table use the best checkpoint; the CSV also records final-epoch results and exact settings for each seed. The three-seed rows show individual seed scores and mean ± sample standard deviation. Single-change ablations were repeated with all three seeds after the initial run.

## Results

Accuracy is shown as percent. Seed columns are ordered 0 / 1 / 2. “Historical regular” is reported only for runs where that separate evaluation was performed.

| Variant | Group regular, seeds 0/1/2; mean ± SD | Group camera-like, seeds 0/1/2; mean ± SD | Historical regular, mean ± SD |
|---|---:|---:|---:|
| Task 3 baseline (LeafNetBN) | 82.3/69.4/79.0; **76.9 ± 6.7** | 59.7/64.5/64.5; **62.9 ± 2.8** | 45.6 ± 8.2 |
| ColorJitter only | 74.2/74.2/74.2; **74.2 ± 0.0** | 67.7/59.7/64.5; **64.0 ± 4.1** | — |
| Dropout p=0.3 only | 74.2/71.0/75.8; **73.7 ± 2.5** | 58.1/58.1/54.8; **57.0 ± 1.9** | — |
| Global average pooling only | 67.7/67.7/71.0; **68.8 ± 1.9** | 56.5/54.8/56.5; **55.9 ± 0.9** | — |
| AdamW, weight decay 1e-4 only | 82.3/69.4/75.8; **75.8 ± 6.5** | 61.3/56.5/64.5; **60.8 ± 4.1** | — |
| Label smoothing 0.1 only | 83.9/74.2/77.4; **78.5 ± 4.9** | 64.5/58.1/61.3; **61.3 ± 3.2** | — |
| Pretrained MobileNet, frozen features | 88.7/90.3/88.7; **89.2 ± 0.9** | 77.4/80.6/83.9; **80.6 ± 3.2** | — |
| Pretrained MobileNet, full fine-tuning | 90.3/77.4/82.3; **83.3 ± 6.5** | 79.0/75.8/80.6; **78.5 ± 2.5** | — |
| MobileNet + ColorJitter + cosine, 30 epochs | 83.9/83.9/88.7; **85.5 ± 2.8** | 80.6/85.5/79.0; **81.7 ± 3.4** | — |
| CNN + ColorJitter + AdamW + smoothing | 75.8/77.4/77.4; **76.9 ± 0.9** | 66.1/59.7/69.4; **65.1 ± 4.9** | 43.9 ± 7.5 |
| MobileNet + ColorJitter + AdamW + smoothing + cosine | 82.3/85.5/87.1; **84.9 ± 2.5** | 82.3/79.0/80.6; **80.6 ± 1.6** | 54.4 ± 2.4 |
| Frozen MobileNet + ColorJitter + cosine, 30 epochs | 88.7/88.7/88.7; **88.7 ± 0.0** | 82.3/83.9/83.9; **83.3 ± 0.9** | 47.5 ± 2.9 |
| Frozen MobileNet + ColorJitter + AdamW + smoothing | 88.7/87.1/88.7; **88.2 ± 0.9** | 85.5/87.1/85.5; **86.0 ± 0.9** | 49.2 ± 2.2 |
| Task 3 baseline, group + historical training | 77.4/79.0/75.8; **77.4 ± 1.6** | 51.6/71.0/61.3; **61.3 ± 9.7** | 81.7 ± 4.3 |
| Recommended combo, group + historical training | 85.5/83.9/85.5; **84.9 ± 0.9** | 79.0/85.5/83.9; **82.8 ± 3.4** | 84.2 ± 2.5 |

## What the changes contributed

- **Pretraining and freezing the feature extractor helped most.** Against the Task 3 baseline, frozen MobileNet improved mean group regular accuracy from 76.9% to 89.2% and camera-like accuracy from 62.9% to 80.6%. The gains are larger than the observed seed variation, though the validation set remains small.
- **Full fine-tuning was less consistent.** Its group accuracy averaged 83.3% and ranged 77.4–90.3%, below frozen features. The small group dataset appears to make updating all pretrained layers less stable.
- **More training, augmentation, and cosine scheduling helped camera-like results modestly.** The 30-epoch full fine-tuning variant averaged 81.7% camera-like versus 78.5% for ordinary full fine-tuning, but its 3.2-point gain is near the seed variation; regular accuracy averaged 85.5%. This bundle did not beat the frozen-feature variants.
- **ColorJitter alone was mixed.** Group accuracy was below baseline; camera-like accuracy averaged 64.0% versus baseline 62.9%, a small gain that does not clearly exceed the measurement noise. It became useful in the selected frozen MobileNet combination.
- **Dropout and global average pooling hurt this CNN.** Their mean regular/camera-like scores were 73.7%/57.0% and 68.8%/55.9%, respectively, versus 76.9%/62.9% for baseline.
- **AdamW and label smoothing alone did not establish a reliable improvement.** AdamW averaged 75.8%/60.8%; label smoothing averaged 78.5%/61.3%. Differences from baseline are small relative to seed spread and validation granularity.
- **The best combination depended on the target domain.** Group-only frozen MobileNet + ColorJitter + AdamW + label smoothing averaged 88.2% regular and 86.0% camera-like. Adding historical training images reduced those to 84.9% and 82.8%, while improving historical validation from 49.2% to 84.2%. The latter is recommended for use across students' phones.
- **Best-epoch selection mattered.** Across baseline seeds, the best regular-validation checkpoints averaged 76.9% regular / 62.9% camera-like; final epochs averaged 72.6% / 59.7%. The selected mixed-data recommendation averages 84.9% / 82.8% at the best epoch, compared with 79.6% / 84.4% at final epoch. Checkpoints were selected on regular group validation, so that metric is optimistic as a selection score; camera-like and historical scores are useful checks, not fully independent test sets.

One validation image changes group accuracy by 1/62 = **1.6 percentage points**. Group class counts are unequal, and the baseline's seed-to-seed regular accuracy standard deviation was 6.7 points. Therefore small differences (roughly a few points) should not be treated as decisive. The stronger frozen-model gains and the historical-data transfer gain are much larger than this resolution and the observed seed spread, but should still be confirmed on more independently photographed leaves.

## Historical photographs: separate domain-transfer comparison

The Task 3 baseline trained on group images alone averaged **45.6%** on the 120 historical validation photos. Training the same baseline on group plus historical training images averaged **81.7%** there, while group regular accuracy was nearly unchanged (76.9% → 77.4%) and camera-like accuracy was similar/slightly lower (62.9% → 61.3%). The selected MobileNet combination shows the same pattern more strongly: group-only historical validation averaged 49.2%; group-plus-historical averaged 84.2%.

These results measure transfer to a separate photo session, students, and phones. They support including historical training images for broad camera use. They do not prove performance on every phone or species appearance. The additional historical camera-like score (87.5%) uses a different, fixed transformation from the group camera-like set, so it is supplemental and is not compared in the main table.

## Run settings and files

All runs use four output classes and 96×96 inputs. The individual CNN ablations change one element at a time from the baseline: ColorJitter `(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04)`; dropout `p=0.3` after the hidden ReLU; global average pooling over the final 6×6 feature map; AdamW with weight decay `1e-4`; or cross-entropy label smoothing `0.1`. These use the same 15 epochs and base Adam learning rate `0.001` except for the optimizer comparison. The CNN+ColorJitter+AdamW+label-smoothing combination also uses 15 epochs.

MobileNet runs use torchvision's pretrained `MobileNet_V3_Small_Weights.DEFAULT` and replace the last classifier layer with four outputs. Frozen-head runs freeze all feature parameters and batch-normalization state; full fine-tuning updates the complete network. The stronger-training run uses full fine-tuning, geometric and color augmentation, Adam at 0.001, `CosineAnnealingLR(T_max=30)`, and 30 epochs. The frozen MobileNet combinations use the same frozen features, geometric/color augmentation and either (a) Adam with cosine schedule for 30 epochs or (b) AdamW `weight_decay=1e-4`, label smoothing `0.1`, no scheduler, 15 epochs. All use batch size 32 and seeds 0–2.

Files added or updated for Task 4:

- `task4_experiments.py` — readable Task 3-based models, training loop, camera-like set creation, evaluation, and checkpoint saving.
- `repeat_ablations.py`, `run_task4_combinations.py`, `run_mixed_mobile.py` — repeated single-change and combination runs.
- `task4_results/results.csv` — every run, seed, settings, best and final scores.
- `task4_results/summary.json` — per-variant means, ranges, standard deviations, and per-seed scores.
- `task4_results/checkpoints/` — best and final PyTorch checkpoint for every run (kept locally; raw training photos and checkpoints are excluded from version control).
- `evaluate_historical.py` — evaluates the exported ONNX model using the app's channel order and normalization.
- `export_recommended.py` — exports the chosen checkpoint, writes the app metadata and expected check-image probabilities, and checks PyTorch/ONNX logit parity.
- `leaf_model.onnx`, `model_info.json`, `check_image.jpg`, `check_expected.json` — course app export bundle.

No cells or explanations in Tasks 1–3 were changed, and nothing under `sources/` was modified. The repository currently has no `REVIEW.md`; this report is in the project root next to the course app files. The phone/camera-app check remains the only requested practical check that could not be performed here.
