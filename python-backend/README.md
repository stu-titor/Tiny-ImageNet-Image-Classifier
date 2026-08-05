# Tiny ImageNet Image Classifier Backend

This directory contains the PyTorch model, Tiny ImageNet training notebook, saved weights, and Flask inference service used by the desktop client.

The deployed network predicts among all 200 Tiny ImageNet classes from 64×64 RGB inputs. The included checkpoint, `trained_net_70.12.pth`, records **70.12% top-1 validation accuracy** and the API returns the five most likely classes.

## Model Summary

| Property | Value |
|---|---|
| Dataset | Tiny ImageNet-200 |
| Classes | 200 |
| Input | 64×64 RGB |
| Included checkpoint | `src/trained_net_70.12.pth` |
| Recorded validation accuracy | **70.12% top-1** |
| API output | Top-5 predictions |
| Training epochs | 300 |
| Training batch size | 512 |

`ImageNeuralNetwork` is configured by the API as `ImageNeuralNetwork(64, 4, 6, 200)`. Its main components are:

- Six convolutional stages with two 3×3 convolutions per stage
- Batch normalization and SiLU activations
- Squeeze-and-excitation channel attention
- Residual connections with progressively increasing stochastic depth
- 2×2 max pooling followed by adaptive global average pooling
- Four fully connected layers with dropout and a 200-unit output layer

## Layout

```text
python-backend/
├── README.md
├── animal.jpg                  # Sample image
├── uni.jpg                     # Sample image
└── src/
    ├── CNN.py                  # Model architecture
    ├── app.py                  # Flask API and deployed inference pipeline
    ├── main.ipynb              # Tiny ImageNet training and evaluation notebook
    ├── trained_net_70.12.pth   # Deployed Tiny ImageNet weights
    └── wnids.txt               # Sorted list of 200 Tiny ImageNet WordNet IDs
```

## Installation

From `python-backend/src`, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the runtime dependencies:

```bash
pip install torch torchvision flask pillow requests timm
```

The training notebook uses the same packages. A CUDA-enabled PyTorch installation is recommended for training but is not required to serve predictions.

## Run the API

Start Flask from the source directory so local imports resolve normally:

```bash
cd python-backend/src
python app.py
```

At startup, the service:

1. Selects CUDA when available, otherwise CPU.
2. Creates the 200-class model and loads `trained_net_70.12.pth`.
3. Loads the 200 WordNet IDs from `wnids.txt`.
4. Listens on `http://localhost:5000`.

## API Usage

### Classify a local file

Send a multipart request whose field name is `image`:

```bash
curl -X POST \
  -F "image=@../animal.jpg" \
  http://localhost:5000/classify/file
```

### Classify an image URL

Send JSON containing the remote image address:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/image.jpg"}' \
  http://localhost:5000/classify/url
```

Both endpoints return a JSON array ordered by confidence:

```json
[
  "prediction #1: class description - 72.34% chance",
  "prediction #2: class description - 11.20% chance",
  "prediction #3: class description - 6.18% chance",
  "prediction #4: class description - 3.92% chance",
  "prediction #5: class description - 2.41% chance"
]
```

Class descriptions are resolved from Tiny ImageNet WordNet IDs with `timm.data.ImageNetInfo`.

## Inference Pipeline

For each request, the API:

1. Opens the image and converts it to RGB.
2. Resizes it to 64×64 pixels.
3. Normalizes it with Tiny ImageNet channel statistics:
   - Mean: `[0.4802, 0.4481, 0.3975]`
   - Standard deviation: `[0.2770, 0.2691, 0.2821]`
4. Runs inference on both the original image and a horizontal flip.
5. Averages the two model outputs and returns the top five softmax scores.

## Training

Training is contained in `src/main.ipynb`. The notebook is designed around a Google Colab workflow: it downloads Tiny ImageNet, reorganizes the validation images into `ImageFolder`-compatible class directories, trains the network, and periodically saves improved checkpoints.

The current recipe includes:

- Random crop with padding, horizontal flip, color jitter, RandAugment, and random erasing
- MixUp or CutMix for the first 285 of 300 epochs
- Cross-entropy loss with `0.1` label smoothing
- SGD with Nesterov momentum, `1e-4` weight decay, and a cosine learning-rate schedule
- Exponential moving-average model weights with `0.999` decay

Before running the notebook outside its original environment, update the dataset and sample-image paths to match your machine. Tiny ImageNet validation images must remain arranged into one subdirectory per WordNet class for `torchvision.datasets.ImageFolder`.
