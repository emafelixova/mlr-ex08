# Leaf Classifier

A phone friendly leaf classifier that runs locally in the browser with ONNX Runtime Web. It uses the rear camera for live predictions and supports photo upload as a fallback.

## Live page

https://emafelixova.github.io/mlr-ex08/

## Model inputs

Preprocessing is loaded from `model_info.json`: center square crop, resize to the configured input size, RGB values scaled to 0–1, per-channel mean and standard deviation, and NCHW tensor layout.

The `check_expected.json` file contains the expected probabilities for the supplied check image.
