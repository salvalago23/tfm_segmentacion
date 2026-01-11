# Skin Lesion Segmentation Web Application

A web application for automatic skin lesion segmentation using deep learning models.

## Features

- **Image Upload**: Upload dermoscopic images for analysis via drag-and-drop or file selection
- **Multiple Models**: Choose from various trained segmentation models (U-Net, Attention U-Net, ResidualU-Net, U-Net++, DeepLabV3+, TransUNet)
- **Real-time Inference**: Get segmentation results with inference time measurements
- **Patient Management**: Create patient profiles to track lesion evolution over time
- **Result History**: Store and compare previous analyses for each patient
- **Visualization**: View original image, segmentation mask, and overlay

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
