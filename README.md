# 📧 Spam Message Classification — MLOps

An end-to-end Machine Learning and MLOps project for classifying SMS messages as **Spam** or **Ham**.

## 🚀 Overview

This project implements a complete and reproducible machine learning pipeline:

**Data Ingestion → Data Preprocessing → TF-IDF → LinearSVC → Model Evaluation**

The project uses **DVC** for pipeline management and experiment tracking, **DVCLive** for metric logging, and **Git/GitHub** for version control.

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TF-IDF
- LinearSVC
- Git & GitHub
- DVC
- DVCLive
- YAML

## 📂 Project Structure

"```text
MLOPS/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
├── reports/
├── src/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── data_feature_engineering.py
│   ├── model_building.py
│   └── model_evaluation.py
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
├── .gitignore
├── .dvcignore
└── README.md
"```

## ⚙️ Configuration

Model and pipeline parameters are maintained in `params.yaml`.

"```yaml
data_ingestion:
  test_size: 0.20

feature_engineering:
  max_features: 3000

model_building:
  C: 1.0
  random_state: 42
"```

## 🔄 DVC Pipeline

The project uses DVC to create a reproducible machine learning pipeline.

### Pipeline stages

1. Data Ingestion
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation

Run the complete pipeline:

"```bash
dvc repro
"```

DVC tracks dependencies, parameters, and outputs using `dvc.yaml` and `dvc.lock`.

## 🧪 Experiment Tracking

DVC experiments allow different model configurations to be tested without manually modifying the parameter file.

### Experiment 1

"```bash
dvc exp run --set-param model_building.C=0.1
"```

### Experiment 2

"```bash
dvc exp run --set-param model_building.C=10
"```

Compare experiments:

"```bash
dvc exp show
"```

DVCLive is used to log evaluation metrics during experiments.

## 📊 Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 98.83% |
| Precision | 99.28% |
| Recall | 91.95% |
| F1 Score | 95.47% |
| ROC-AUC | 98.94% |

Metrics are stored in `reports/metrics.json`.

----

## 🎯 MLOps Concepts Demonstrated

- Modular machine learning pipeline
- Data preprocessing
- TF-IDF feature engineering
- LinearSVC classification
- YAML-based parameter management
- DVC pipeline management
- Reproducible ML workflows
- DVC experiment tracking
- DVCLive metric logging
- Git and GitHub version control

## 🔮 Future Improvements

- Dockerization
- CI/CD integration
- FastAPI model deployment
- Cloud-based DVC remote storage
- Model monitoring
- Automated hyperparameter tuning

## 👨‍💻 Author

**Nikhil Garg**

Machine Learning & MLOps Project
