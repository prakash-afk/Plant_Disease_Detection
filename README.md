# Plant Disease Detection

A deep learning project for classifying plant leaf diseases from images using TensorFlow, FastAPI, and Streamlit.

The project covers the full workflow:

- downloading the PlantVillage dataset from Kaggle
- preparing and visualizing the training data
- training a transfer learning model with MobileNetV2
- evaluating model performance on validation data
- serving predictions through a FastAPI backend
- using a Streamlit UI for interactive leaf image analysis

## Project Overview

This repository is designed for plant disease classification on leaf images from the PlantVillage dataset. The trained model is saved in Keras format and used by a local prediction API and frontend.

Current model artifact:

- `models/plant_disease_model.keras`

Current label mapping:

- `models/class_indices.json`

## Features

- TensorFlow/Keras training pipeline
- MobileNetV2-based transfer learning model
- image augmentation for training
- evaluation script with classification report and confusion matrix
- FastAPI prediction endpoint
- Streamlit interface for uploading images and viewing top predictions
- treatment suggestion mapping for predicted classes
- raw confidence display without forced rounding

## Repository Structure

```text
Plant_Disease/
|-- app.py
|-- download_data.py
|-- evaluate.py
|-- pre_processing.py
|-- requirements.txt
|-- save_class_indices.py
|-- streamlit_app.py
|-- train.py
|-- data/
|-- models/
```

## Main Scripts

- `download_data.py`: Downloads and extracts the dataset from Kaggle.
- `pre_processing.py`: Creates data generators and saves a sample visualization.
- `train.py`: Trains the MobileNetV2-based classifier and saves the best model.
- `save_class_indices.py`: Saves class-to-index mapping to `models/class_indices.json`.
- `evaluate.py`: Loads the trained model and generates evaluation metrics and plots.
- `app.py`: FastAPI backend for image prediction.
- `streamlit_app.py`: Streamlit frontend for interacting with the model.

## Dataset

This project uses the Kaggle PlantVillage dataset:

- Dataset: `vipoooool/new-plant-diseases-dataset`

Expected dataset layout after extraction:

```text
data/PlantVillage/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/
|-- train/
|-- valid/
```

## Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure Kaggle credentials

Create a `.env` file in the project root:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

## End-to-End Workflow

### 1. Download and extract the dataset

```powershell
python download_data.py
```

### 2. Prepare and visualize sample data

```powershell
python pre_processing.py
```

This generates sample visualizations such as:

- `data/sample_images.png`

### 3. Save class indices

```powershell
python save_class_indices.py
```

### 4. Train the model

```powershell
python train.py
```

Training uses:

- `MobileNetV2` backbone
- frozen base model in phase 1
- partial fine-tuning in phase 2
- `ModelCheckpoint`
- `EarlyStopping`
- `ReduceLROnPlateau`

Saved model:

- `models/plant_disease_model.keras`

### 5. Evaluate the model

```powershell
python evaluate.py
```

Evaluation outputs include:

- validation accuracy
- `models/classification_report.txt`
- `models/confusion_matrix.png`

## Running the App

### Start the FastAPI backend

```powershell
python -m uvicorn app:app --reload
```

Backend endpoint:

- `POST /predict`

### Start the Streamlit frontend

```powershell
streamlit run streamlit_app.py
```

## Prediction Output

The API returns:

- `predicted_class`
- `confidence`
- `top_predictions`

The Streamlit app displays:

- predicted disease or healthy class
- confidence score
- top 3 predictions
- treatment suggestion

## Model Notes

- Input image size for training and inference: `224 x 224`
- Model format: `.keras`
- Confidence values are currently shown as raw float percentages

## Known Limitations

- The current system is a closed-set classifier.
- It always predicts one of the known PlantVillage classes.
- If you upload a non-leaf image or an out-of-distribution image, the model may still return a plant disease class.
- Predictions should be treated as project output, not as a medical or agricultural diagnosis.

## To Be Improved

- Add open-set or unknown-class rejection so non-leaf images can be rejected safely.
- Add stronger out-of-distribution detection.
- Improve preprocessing and input validation before inference.
- Add model versioning and reproducible training metadata.
- Add automated tests for the API and frontend workflow.
- Improve deployment readiness for cloud hosting.

## Suggested Workflow

- Train in Google Colab if GPU access is easier there.
- Copy the trained `.keras` model back into the local `models/` folder.
- Use VS Code locally for evaluation, API serving, and Streamlit inference.

## Disclaimer

This project is for learning, experimentation, and demonstration purposes. For real agricultural decisions, consult an agronomist, plant pathologist, or domain expert.
