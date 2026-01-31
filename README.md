# Image Convolution Engine

A full-stack demo that applies 2D convolution kernels to images without using OpenCV shortcuts. The project ships with a FastAPI backend for the convolution engine and a Streamlit frontend that lets you preview pre-built or custom kernels.

## Project Structure

```
image-convolution-engine/
├── backend/
│   ├── convolution.py      # Core convolution implementation
│   ├── kernels.py          # Pre-defined kernels and validation helpers
│   └── main.py             # FastAPI application
├── frontend/
│   └── app.py              # Streamlit UI
├── requirements.txt        # Shared dependencies
└── README.md
```

## Prerequisites

- Python 3.12 (recommended). Pillow 10.x does **not** ship wheels for Python 3.13, so stick with 3.12 to avoid build errors.
- macOS / Linux / WSL (instructions assume a POSIX shell).

Install Python 3.12 using your preferred tool (e.g. `pyenv`, `conda`).

## Setup

```bash
# Clone the repository (replace with your fork/URL)
git clone <repo-url>
cd image-convolution-engine

# Create and activate a Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> If you accidentally use Python 3.13, `pip` will try compiling Pillow 10.x from source and fail with `KeyError('__version__')`. Switch back to Python 3.12 or lower.

## Running the Backend

```bash
# From the project root (venv active)
cd backend
uvicorn main:app --reload
```

- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health check: `GET /health`

### Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/kernels/list` | Returns pre-built kernels |
| `POST` | `/kernels/custom` | Validates a custom kernel matrix |
| `POST` | `/transform/convolution` | Applies a kernel to an uploaded image |

## Running the Frontend

In a second terminal (keep the backend running):

```bash
cd image-convolution-engine  # project root
source .venv/bin/activate    # ensure you are in the same venv
streamlit run frontend/app.py
```

The UI launches at [http://localhost:8501](http://localhost:8501).

### Using the UI

1. **Pre-built Kernels**
   - Pick a kernel from the selector to preview its matrix.
   - Upload an image (`≤ 2 MB`).
   - Click **“Apply Convolution”** to view and download the result.

2. **Custom Kernel**
   - Switch “Select Mode” to **Custom Kernel**.
   - Enter a JSON matrix (must be square, odd-sized between 3×3 and 11×11).
     ```json
     [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
     ```
   - Upload an image (`≤ 2 MB`).
   - The app validates the kernel through the `/kernels/custom` API before enabling **Apply**.
   - After processing, download the transformed image.

> The frontend sends the original uploaded bytes to the backend. Re-saving large JPEGs as PNGs can inflate size—if you still hit the 2 MB limit, resize or compress the image prior to upload.

## Customisation

- Add new kernels in `backend/kernels.py` and update the dictionary returned by `list_kernels()`.
- Tune convolution padding (reflect/edge/constant) by adjusting `ConvolutionEngine(padding_mode=...)` in `backend/main.py`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: fastapi` | Activate the correct virtualenv (`source .venv/bin/activate`) before running the backend. |
| Pillow build error on install | Use Python 3.12 and reinstall dependencies. |
| Streamlit cannot connect to API | Ensure the backend is running at `http://localhost:8000`. If using another host/port, change `API_URL` in `frontend/app.py`. |
| “File too large. Maximum size is 2MB” | Verify the original file size; if still under 2 MB, recompress or resize before upload. |

# Image-Kernel-Convolution-Engine
