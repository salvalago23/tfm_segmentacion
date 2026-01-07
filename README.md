## Prerrequisitos
- Git LFS (para poder descargar los modelos)

## Descarga del proyecto
- git lfs install
- git clone https://github.com/yourname/your-repo.git

# Instalación (en el directorio raíz del proyecto)
- conda env create -f environment.yml
- conda activate tfm_segmentacion_env

- python3 download_dataset.py


## ARCHIVOS DESCARGADOS (ISIC 2018)
El dataset completo ocupa ~13 GB

- ISIC2018 Training Input (10.46 GB) → 2594 imágenes training
- ISIC2018 Training GroundTruth (33 MB) → 2594 máscaras training
- ISIC2018 Validation Input (228 MB) → 100 imágenes validation
- ISIC2018 Validation GroundTruth (1 MB) → 100 máscaras validation
- ISIC2018 Test Input (2.26 GB) → 1000 imágenes test
- ISIC2018 Test GroundTruth (11 MB) → 1000 máscaras test


### Todo el EDA, entrenamiento de modelos y análisis de los resultados de los mismos se pueden encontrar en los 3 Jupyter Notebook del directorio ./notebooks

### Para ejecutar la app, basta con moverse al directorio ./app y ejecutar el archivo bash run.sh







