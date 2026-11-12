# Leaf classifier review

Review target: the checked-in `index.html`, `leaf_model.onnx`, `model_info.json`, `check_expected.json`, and `check_image.jpg`, plus the supplied `Kopie sešitu ex08_student.ipynb`.

## 1. Runs on a phone

**What I did:** Opened the public Pages URL, `https://emafelixova.github.io/mlr-ex08/`, and inspected the camera flow in `index.html`. The page requests `facingMode: environment`, displays the stream in a `playsinline` video, and classifies frames repeatedly after the model is ready.

**What I found:** The Pages page and model load in the browser. I have not verified live-camera access on an iPhone. The required phone screenshot with a leaf and probabilities has not been supplied, so this requirement is still **pending**.

**What was fixed:** No phone-specific change was made without a phone test. Add the phone screenshot here after opening the Pages URL and confirming that camera permission and live predictions work.

**Evidence:** _Pending phone screenshot._

## 2. Check image probabilities

**What I did:** Compared `check_image.jpg` with `check_expected.json` using the current ONNX model and the page's center-crop, resize, and normalization steps. I also attempted to select the image through the live page's upload control, but the browser's file-upload permission check declined the action; I did not retry it through another route.

**What I found:** The local ONNX inference agrees with the expected values to three decimals. The actual values below are from local inference with the page's preprocessing; they are not represented as a completed browser-upload result.

| Class | Inference probability | `check_expected.json` | To three decimals |
| --- | ---: | ---: | --- |
| apple | 0.538040638 | 0.538040578 | 0.538 = 0.538 |
| cherry | 0.427219599 | 0.427219361 | 0.427 = 0.427 |
| chestnut | 0.025417337 | 0.025417605 | 0.025 = 0.025 |
| maple | 0.009322396 | 0.009322453 | 0.009 = 0.009 |

**What was fixed:** No code change was needed for the numerical comparison. The on-page upload step remains unverified because the browser blocked file selection.

## 3. Preprocessing matches the model and notebook

**What I did:** Read `cropDraw` and `classify` in `index.html`, `model_info.json`, and the Task 2 data and transform cells in the supplied notebook (cells 37, 40, and 49).

**What I found:** The page takes a centered square from each frame or uploaded image, resizes it to `input_size` (96 × 96), reads canvas RGB bytes, divides each channel by 255, and standardizes R, G, and B using the `mean` and `std` loaded from `model_info.json`. It stores the channels in separate contiguous planes in R, G, B order, then creates a float tensor with shape `[1, 3, 96, 96]`.

The notebook says the Task 2 training images were center-cropped and resized to 96 × 96 when the dataset was prepared. Its Task 2 transform is `ToTensor()` followed by `Normalize(mean, std)`: `ToTensor()` scales RGB bytes to 0–1 and changes HWC to CHW. The notebook's exported metadata uses the same input size, class list, mean, and standard deviation. The page's inference preprocessing therefore matches the Task 2 model pipeline. The notebook's random crop, flips, and rotation belong to the separate Task 3 training augmentation and are not applied at inference.

**What was fixed:** No preprocessing change was needed; the inspected page and metadata agree with the notebook's inference-time transform.

## 4. Image privacy and page requests

**What I did:** Inspected the page's network references and image-handling code.

**What I found:** Camera frames and selected photos are drawn to a browser canvas and passed to ONNX Runtime Web in the browser. The page has no image-upload endpoint or remote inference call, so it does not send the image to the model host or CDN. The page requests:

- `https://emafelixova.github.io/mlr-ex08/` for the page and same-origin model files;
- `./model_info.json` and `./leaf_model.onnx` from that Pages site, to configure and run the classifier;
- `https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js` for ONNX Runtime Web;
- WebAssembly runtime assets from `https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/` so inference can run in the browser.

The QR code is embedded in the HTML as a data image and makes no QR-service request. These requests transfer the page, model metadata, model, and runtime code; they do not contain the camera frame or selected photo. The hosting and CDN services still receive ordinary page and asset requests from the browser.

**What was fixed:** No image-transfer code was found that needed fixing. The QR was embedded directly in the page, so it does not rely on a third-party QR-image service.
