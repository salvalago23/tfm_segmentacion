RESUMEN COMPLETO DEL PROYECTO TFM - SEGMENTACIÓN DE IMÁGENES MÉDICAS
📁 ESTRUCTURA DEL PROYECTO COMPLETA
text
tfm_segmentacion/
│
├── data/                               # Datos
│   ├── raw/                            # Datos originales descargados
│   │   ├── isic2018_train/             # Training oficial (2594 imágenes)
│   │   │   ├── images/                 # ISIC_XXXXXXX.jpg
│   │   │   └── masks/                  # ISIC_XXXXXXX_segmentation.png
│   │   ├── isic2018_val/               # Validation oficial (100 imágenes)
│   │   │   ├── images/
│   │   │   └── masks/
│   │   └── isic2018_test/              # Test oficial (1000 imágenes)
│   │       └── images/                 # Sin máscaras públicas
│   │
│   └── processed/                      # Datos preprocesados (por crear)
│       ├── train/
│       ├── val/
│       └── test/
│
├── notebooks/                          # Análisis exploratorio
│   ├── 01_eda.ipynb                    # EDA original (HAM10000 + ISIC)
│   └── 01_eda_isic2018.ipynb           # EDA completo (Train/Val/Test)
│
├── src/                                # Código fuente
│   ├── data_preparation/
│   │   ├── __init__.py
│   │   ├── data_loader.py              # MedicalDataLoader (PyTorch)
│   │   ├── dataset.py                  # ISIC2018Dataset
│   │   └── preprocessing.py            # MedicalImagePreprocessor
│   │
│   ├── models/                         # (POR IMPLEMENTAR)
│   ├── training/                       # (POR IMPLEMENTAR)
│   ├── evaluation/                     # (POR IMPLEMENTAR)
│   └── utils/                          # (POR IMPLEMENTAR)
│
├── scripts/                            # Scripts ejecutables
│   ├── organize_isic2018.py            # Organiza archivos descargados
│   ├── preprocess_dataset.py           # Preprocesa imágenes
│   ├── verify_preprocessing.py         # Verifica preprocesamiento
│   └── test_dataloader.py              # Prueba DataLoader
│
├── config/                             # Configuraciones
│   └── data_config.yaml                # Config data (plantilla)
│
├── eda_results/                        # Resultados del EDA
│   ├── dataset_comparison.csv
│   ├── preprocessing_recommendations.csv
│   └── detailed_stats.json
│
├── downloads/                          # Archivos ZIP descargados (opcional)
├── requirements.txt                    # Dependencias
└── README.md                           # Documentación

📦 ARCHIVOS DESCARGADOS (ISIC 2018)
Dataset completo descargado:
ISIC2018_Task1-2_Training_Input.zip (10.46 GB) → 2594 imágenes training

ISIC2018_Task1_Training_GroundTruth.zip (33 MB) → 2594 máscaras training

ISIC2018_Task1-2_Validation_Input.zip (228 MB) → 100 imágenes validation

ISIC2018_Task1_Validation_GroundTruth.zip (1 MB) → 100 máscaras validation

ISIC2018_Task1-2_Test_Input.zip (2.26 GB) → 1000 imágenes test

ISIC2018_Task1_Test_GroundTruth.zip → 1000 máscaras test

Total descargado: ~13 GB
