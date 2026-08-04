# Waste Segregation AI

An educational computer-vision prototype that classifies waste into recyclable, compost, and landfill categories. The project combines a TensorFlow image classifier with Streamlit interfaces for image prediction and project storytelling.

## Features

- Three-class waste classification: recyclable, compost, and landfill
- Image upload and webcam prediction in `app/streamlit_app.py`
- Confidence scores, class probabilities, and disposal tips
- Training pipeline with augmentation, validation, checkpoints, and reports
- Dashboard showing collection counts, trends, and waste-mix visualizations

## Repository layout

- `app/streamlit_app.py` — interactive prediction app
- `dashboard/dashboard.py` — Streamlit project dashboard
- `scripts/train_model.py` — model training and evaluation
- `data/raw/` — images organized by category
- `models/` — trained model, class mapping, logs, and evaluation artifacts
- `docs/` — proposal, methodology, and final report

## Requirements

Python 3.10+ with TensorFlow, OpenCV, NumPy, Streamlit, pandas, Plotly, scikit-learn, Matplotlib, and Seaborn.

## Run the apps

```bash
pip install tensorflow opencv-python numpy streamlit pandas plotly scikit-learn matplotlib seaborn
streamlit run app/streamlit_app.py
streamlit run dashboard/dashboard.py
```

The prediction app expects `models/best_model.keras` and `models/class_indices.json`. Add appropriately labelled images under `data/raw/<category>/` before training.

## Train the model

```bash
python scripts/train_model.py
```

This is a school capstone prototype. Predictions should be treated as educational guidance rather than authoritative waste-disposal advice.
