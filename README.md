## Prerrequisitos
- Git LFS (install from https://git-lfs.github.com)

## Instalación
git lfs install  # Para descargar los modelos
git clone https://github.com/yourname/your-repo.git

# Una vez descargado, ejecutar desde dentro del directorio ./tfm_segmentacion
conda env create -f environment.yml
conda activate tfm_segmentacion_env

python3 download_dataset.py


ARCHIVOS DESCARGADOS (ISIC 2018)
Dataset completo descargado:
ISIC2018_Task1-2_Training_Input.zip (10.46 GB) → 2594 imágenes training

ISIC2018_Task1_Training_GroundTruth.zip (33 MB) → 2594 máscaras training

ISIC2018_Task1-2_Validation_Input.zip (228 MB) → 100 imágenes validation

ISIC2018_Task1_Validation_GroundTruth.zip (1 MB) → 100 máscaras validation

ISIC2018_Task1-2_Test_Input.zip (2.26 GB) → 1000 imágenes test

ISIC2018_Task1_Test_GroundTruth.zip → 1000 máscaras test

Total descargado: ~13 GB


