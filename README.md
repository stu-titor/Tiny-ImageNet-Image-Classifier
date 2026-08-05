# Tiny ImageNet Image Classifier

A full-stack image classifier for the 200-class Tiny ImageNet dataset. The project combines a custom PyTorch convolutional network and Flask inference API with a Windows WPF desktop client.

The included model accepts RGB images, resizes them to 64×64 pixels, and returns its top five predicted classes. The packaged checkpoint records **70.12% validation accuracy** (*~71% with TTA*).

## Features

- Custom CNN with residual connections, squeeze-and-excitation blocks, stochastic depth, and global average pooling
- 200 Tiny ImageNet output classes identified through WordNet IDs
- Top-5 inference with horizontal-flip test-time augmentation
- Flask endpoints for local file uploads and remote image URLs
- WPF desktop interface with local image preview and an IBM-inspired workstation design
- Automatic CUDA inference when a compatible GPU is available, with CPU fallback

## Project Structure

```text
.
├── desktop-client/             # .NET 10 WPF desktop application
│   ├── MainWindow.xaml         # Tiny ImageNet classification console UI
│   ├── MainWindow.xaml.cs      # File selection, preview, and API calls
│   └── CifarInterface.csproj
├── python-backend/
│   ├── src/
│   │   ├── CNN.py              # PyTorch model definition
│   │   ├── app.py              # Flask inference API
│   │   ├── main.ipynb          # Tiny ImageNet training notebook
│   │   ├── trained_net_70.12.pth
│   │   └── wnids.txt           # The 200 supported WordNet class IDs
│   └── README.md               # Backend and training documentation
└── desktop-client.slnx
```

## Requirements

### Backend

- Python 3.10 or newer
- PyTorch and torchvision
- Flask
- Pillow
- Requests
- timm

### Desktop client

- Windows
- .NET 10 SDK with WPF support

## Quick Start

### 1. Start the inference API

From the repository root:

```bash
cd python-backend/src
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the dependencies and start Flask:

```bash
pip install torch torchvision flask pillow requests timm
python app.py
```

The API listens on `http://localhost:5000`.

### 2. Start the desktop client

In a second terminal, from the repository root:

```bash
dotnet run --project desktop-client/CifarInterface.csproj
```

Choose an image file or enter an image URL, then select **Run Classification**. Local files appear in the image monitor as soon as they are selected.

## API Endpoints

| Method | Endpoint | Input |
|---|---|---|
| `POST` | `/classify/file` | Multipart form field named `image` |
| `POST` | `/classify/url` | JSON object containing a `url` field |

Both endpoints return a JSON array containing five prediction strings ordered from highest to lowest confidence. See [python-backend/README.md](python-backend/README.md) for examples and model details.

## Notes

- The desktop client expects the Flask service at `http://localhost:5000`.
- The included `trained_net_70.12.pth` checkpoint must remain beside `app.py`.
- Training is documented in the Colab-oriented notebook at `python-backend/src/main.ipynb`.
- Generated folders such as `.vs/`, `bin/`, `obj/`, `__pycache__/`, and notebook checkpoints are ignored by Git.
