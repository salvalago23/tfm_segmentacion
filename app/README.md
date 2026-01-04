# Skin Lesion Segmentation Web Application

A web application for automatic skin lesion segmentation using deep learning models.

## Features

- **Image Upload**: Upload dermoscopic images for analysis via drag-and-drop or file selection
- **Multiple Models**: Choose from various trained segmentation models (U-Net, Attention U-Net, ResidualU-Net, U-Net++, DeepLabV3+, TransUNet)
- **Real-time Inference**: Get segmentation results with inference time measurements
- **Patient Management**: Create patient profiles to track lesion evolution over time
- **Result History**: Store and compare previous analyses for each patient
- **Visualization**: View original image, segmentation mask, and overlay

## Requirements

- Python 3.8+
- Trained model files in `notebooks/experiments/notebook_experiments/`
- Dependencies from `requirements.txt`

## Installation

1. Make sure you have the main project dependencies installed:
   ```bash
   pip install -r ../requirements.txt
   ```

2. Install app-specific dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Using the run script (recommended)

```bash
chmod +x run.sh
./run.sh
```

### Option 2: Manual start

```bash
cd backend
PYTHONPATH=../.. python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open http://localhost:8000 in your browser.

## Structure

```
app/
├── backend/
│   ├── main.py          # FastAPI server with all endpoints
│   └── data/            # Auto-created directory for patient data
│       ├── patients.json
│       └── results/
├── frontend/
│   └── index.html       # Single-page web interface
├── requirements.txt     # App-specific Python dependencies
├── run.sh              # Startup script
└── README.md           # This file
```

## API Endpoints

### Health & Info
- `GET /api/health` - Health check and device info
- `GET /api/models` - List available models

### Patients
- `POST /api/patients` - Create new patient
- `GET /api/patients` - List all patients
- `GET /api/patients/{id}` - Get patient details
- `DELETE /api/patients/{id}` - Delete patient
- `GET /api/patients/{id}/results` - Get patient's analysis history

### Prediction
- `POST /api/predict` - Run segmentation on uploaded image
  - Parameters: `file` (image), `model` (model key), `patient_id`
  - Returns: Original image, segmentation mask, overlay, inference time, lesion coverage

## Notes

- Patient data is stored in a simple JSON file (`backend/data/patients.json`)
- Images are stored as base64 in the database (suitable for small-scale use)
- For production use, consider using a proper database and file storage system
