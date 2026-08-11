# Contributing to Waste Segregation AI

Thank you for your interest in contributing! This is a Grade 12 capstone project, and we welcome improvements from the community.

## How to Contribute

### Reporting Bugs
1. Check if the issue already exists in [Issues](https://github.com/aaronsineshdxb/waste-segregation-ai-/issues)
2. If not, create a new issue using the **Bug Report** template
3. Include steps to reproduce, expected vs actual behavior, and environment details

### Suggesting Features
1. Check existing issues and discussions
2. Create a new issue using the **Feature Request** template
3. Describe the feature, its benefits, and any implementation ideas

### Pull Requests
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes following the coding standards below
4. Run tests and linting: `black --check . && flake8 .`
5. Commit with clear messages: `git commit -m "feat: add new feature"`
6. Push to your fork and open a PR against `main`

## Coding Standards

### Python Style
- Follow [PEP 8](https://pep8.org/) with 100-char line length
- Use type hints where practical
- Format with `black` (run `black .` before committing)
- Lint with `flake8` (run `flake8 .` before committing)

### Commit Messages
Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` code restructuring
- `test:` adding tests
- `chore:` maintenance tasks

### Project Structure
```
├── app/              # Streamlit prediction app
├── dashboard/        # Streamlit dashboard
├── scripts/          # Training and data collection scripts
├── data/raw/         # Training images (gitignored)
├── models/           # Trained models and artifacts (gitignored)
├── docs/             # Documentation
├── config/           # Configuration files
└── .github/          # CI/CD and templates
```

## Development Setup

```bash
# Clone and setup
git clone https://github.com/aaronsineshdxb/waste-segregation-ai-.git
cd waste-segregation-ai-

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

## Running the Project

### Data Collection
```bash
python scripts/collect_data.py --category all --num 50
```

### Training
```bash
# With defaults
python scripts/train_model.py

# With custom config
python scripts/train_model.py --config config/training_config.yaml

# Evaluate existing model
python scripts/train_model.py --eval-only
```

### Apps
```bash
# Prediction app
streamlit run app/streamlit_app.py

# Dashboard
streamlit run dashboard/dashboard.py
```

## Testing

Before submitting a PR, ensure:
- [ ] Code passes `black --check .`
- [ ] Code passes `flake8 .`
- [ ] Type checking passes: `mypy scripts/ app/ dashboard/ --ignore-missing-imports`
- [ ] Training script runs without errors: `python scripts/train_model.py --help`
- [ ] All imports work: `python -c "import scripts.train_model; import app.streamlit_app; import dashboard.dashboard"`

## Areas for Contribution

- **Data**: Collect and contribute real waste images from schools
- **Model**: Experiment with architectures, hyperparameters, augmentation
- **UI/UX**: Improve Streamlit apps, add accessibility features
- **Documentation**: Enhance docs, add tutorials, translate
- **Deployment**: Docker, cloud deployment, mobile app
- **Testing**: Add unit tests, integration tests

## Code of Conduct

Be respectful, inclusive, and constructive. This is an educational project - help others learn!

## Questions?

Open a discussion or issue. We're happy to help!