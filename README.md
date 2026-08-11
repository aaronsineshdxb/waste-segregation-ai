# Waste Segregation AI

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/aaronsineshdxb/waste-segregation-ai-/actions/workflows/ci.yml/badge.svg)](https://github.com/aaronsineshdxb/waste-segregation-ai-/actions/workflows/ci.yml)

An educational computer-vision prototype that classifies waste into **recyclable**, **compost**, and **landfill** categories. Built as a Grade 12 capstone project combining TensorFlow/Keras transfer learning with Streamlit interfaces for real-time prediction and project storytelling.

## Features

- **Three-class waste classification**: recyclable, compost, and landfill
- **Real-time prediction**: Webcam and image upload in `app/streamlit_app.py`
- **Confidence scores**, class probabilities, and disposal tips
- **Training pipeline** with augmentation, validation, checkpoints, TensorBoard, and evaluation reports
- **Dashboard** showing collection counts, trends, and waste-mix visualizations
- **Configurable training** via YAML config or CLI arguments
- **Data collection script** for gathering real school waste images

## Repository Layout

```
├── app/
│   └── streamlit_app.py       # Interactive prediction app (webcam + upload)
├── dashboard/
│   └── dashboard.py           # Streamlit project dashboard
├── scripts/
│   ├── train_model.py         # Model training and evaluation
│   ├── collect_data.py        # Webcam data collection
│   └── validate_json.py       # JSON validation for CI
├── config/
│   └── training_config.yaml   # Training hyperparameters
├── data/raw/                  # Training images by category (gitignored)
├── models/                    # Trained models, logs, plots (gitignored)
├── docs/                      # Proposal, methodology, final report
├── .github/                   # CI/CD, issue/PR templates
├── requirements.txt           # Pinned dependencies
├── pyproject.toml             # Modern Python packaging
├── .pre-commit-config.yaml    # Pre-commit hooks
├── .gitignore                 # Git ignore rules
├── LICENSE                    # MIT License
├── CONTRIBUTING.md            # Contribution guidelines
└── README.md                  # This file
```

## Requirements

- Python 3.10+
- TensorFlow 2.15+
- OpenCV, NumPy, Streamlit, Pandas, Plotly, scikit-learn, Matplotlib, Seaborn

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/aaronsineshdxb/waste-segregation-ai-.git
cd waste-segregation-ai-

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### 2. Collect Training Data (Recommended)

Collect real waste images from your school environment:

```bash
# Collect 50 images per category (150 total)
python scripts/collect_data.py --category all --num 50

# Or collect for a specific category
python scripts/collect_data.py --category recyclable --num 100
```

Images are saved to `data/raw/<category>/` with timestamps.

### 3. Train the Model

```bash
# Train with defaults (MobileNetV2, 30+15 epochs)
python scripts/train_model.py

# Train with custom config
python scripts/train_model.py --config config/training_config.yaml

# Train with EfficientNetB0
python scripts/train_model.py --base-model EfficientNetB0 --epochs 40 --fine-tune-epochs 20

# Evaluate existing model only
python scripts/train_model.py --eval-only
```

Outputs saved to `models/`:
- `best_model.keras` - Best validation accuracy model
- `final_model.keras` - Final epoch model
- `class_indices.json` - Category to index mapping
- `training_log.csv` - Per-epoch metrics
- `training_history.png` - Training curves
- `confusion_matrix.png` - Validation confusion matrix
- `evaluation_report.json` - Detailed metrics

### 4. Run the Apps

**Prediction App (Webcam + Upload):**
```bash
streamlit run app/streamlit_app.py
```

**Dashboard (Trends + Storytelling):**
```bash
streamlit run dashboard/dashboard.py
```

## Configuration

Training can be configured via `config/training_config.yaml` or CLI arguments:

```yaml
# config/training_config.yaml
data_dir: "data/raw"
model_dir: "models"
img_size: [224, 224]
batch_size: 32
base_model_name: "MobileNetV2"  # MobileNetV2, EfficientNetB0, ResNet50
use_pretrained: true
epochs: 30
fine_tune_epochs: 15
initial_lr: 0.001
fine_tune_lr: 0.00001
validation_split: 0.2
```

CLI args override config file values:
```bash
python scripts/train_model.py --epochs 50 --batch-size 64 --base-model EfficientNetB0
```

## Model Architecture

- **Base**: Pre-trained MobileNetV2 / EfficientNetB0 / ResNet50 (ImageNet weights)
- **Head**: GlobalAveragePooling2D → BatchNorm → Dropout(0.3) → Dense(256, relu) → BatchNorm → Dropout(0.2) → Dense(3, softmax)
- **Training**: Two-phase transfer learning
  1. Frozen base, train head (epochs 1-N)
  2. Unfreeze last 20 base layers, fine-tune with 100x lower LR (epochs N+1 to N+M)
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard

## Evaluation Metrics

The training script reports:
- Accuracy, Precision, Recall, AUC (per-class and macro/weighted averages)
- Confusion matrix (saved as PNG)
- Classification report (saved as JSON)
- Training history plots (accuracy, loss, precision, recall, AUC, LR)

## Project Documentation

- [Capstone Proposal](docs/capstone_proposal.md) - Project overview and objectives
- [Methodology](docs/methodology.md) - Data Science Methodology (10 steps per AI syllabus)
- [Final Report](docs/capstone_final_report.md) - Complete project documentation

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push/PR:
- Linting (flake8, black)
- Type checking (mypy)
- Import verification
- Training script validation
- Config/JSON validation
- Security scanning (pip-audit, trufflehog)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Bug reports and feature requests
- Pull request process
- Coding standards (black, flake8, mypy)
- Commit message conventions
- Development setup

## Future Improvements

- [ ] Collect more real school waste images
- [ ] Improve model accuracy with more data and hyperparameter tuning
- [ ] Add more waste categories (e-waste, hazardous, etc.)
- [ ] Deploy on mobile devices (TensorFlow Lite)
- [ ] Track school waste reduction over time
- [ ] Add YOLO-based object detection for multiple items
- [ ] Dockerize for easy deployment
- [ ] Add unit tests and increase coverage

## Team

- **Aaron** - Project planning, documentation, model coordination
- **Nishanth** - Data collection support and testing
- **Sidardh** - Model experimentation and app support
- **Safa** - Dashboard and presentation support
- **Sidhiksha** - Research, validation, and documentation support

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*This is a Grade 12 capstone prototype. Predictions should be treated as educational guidance rather than authoritative waste-disposal advice.*