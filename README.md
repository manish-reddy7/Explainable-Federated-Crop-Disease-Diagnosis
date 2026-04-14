# Explainable Federated Crop Disease Diagnosis (XFedCrop)

XFedCrop is a PyTorch-based crop disease diagnosis project with:

- **Centralized training** baseline (ResNet-18)
- **In-process federated simulation** (FedAvg / FedProx)
- **Explainability** (Grad-CAM, LIME, SHAP utilities)
- **FastAPI inference backend** with prediction and explanation endpoints
- **React frontend dashboard** served by the API

---

## Project Structure

- `api/` – FastAPI app and inference endpoints
- `models/` – model definition and training loops
- `fl_simulation/` – local federated simulation
- `explainability/` – Grad-CAM, LIME, SHAP helpers
- `data/` – partition/splitting utilities
- `checkpoints/` – trained model artifacts and reports
- `agri-ai-insights-main/` – React frontend (Vite)

---

## Requirements

- Python 3.10+
- Node.js 18+
- (Optional) CUDA-enabled GPU

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd agri-ai-insights-main
npm install
```

---

## Build Frontend

The FastAPI root route serves `agri-ai-insights-main/dist/index.html`, so build UI first:

```bash
cd agri-ai-insights-main
npm run build
```

---

## Run the Project

From repository root (`XFedCrop/`):

```bash
python -m uvicorn api.app:app --app-dir . --host 0.0.0.0 --port 8000
```

Open:

- App UI: `http://127.0.0.1:8000/`
- API status: `http://127.0.0.1:8000/status`

### Optional checkpoint override

By default, API loads:

- `checkpoints/central_resnet18_best.pt`

You can override with:

```bash
# Windows PowerShell
$env:XFEDCROP_CHECKPOINT="C:/path/to/model.pt"
```

---

## API Endpoints

- `GET /status` – service + model status
- `POST /predict` – image classification (multipart file)
- `POST /explain/gradcam` – Grad-CAM overlay image (PNG)

---

## Training

### Centralized baseline

```bash
python -m models.train_central --parent_dir "archive (4)/PlantVillage" --epochs 15 --batch_size 32 --out_dir checkpoints
```

Alternative split usage:

```bash
python -m models.train_central --data_dir "archive (4)/PlantVillage/train" --val_dir "archive (4)/PlantVillage/val"
```

### Federated simulation (FedAvg / FedProx)

```bash
python -m fl_simulation.local_federated --data_dir "archive (4)/PlantVillage/train" --clients 5 --rounds 20 --alpha 0.0
```

For FedProx:

```bash
python -m fl_simulation.local_federated --data_dir "archive (4)/PlantVillage/train" --clients 5 --rounds 20 --proximal_mu 0.01
```

---

## Testing

```bash
pytest -q
```

---

## Notes

- Ensure frontend build artifacts exist before opening `/` route.
- If `model_loaded` is `false` on `/status`, verify checkpoint path and file presence.
- `.gitignore` is configured to exclude logs, caches, and local environments.
