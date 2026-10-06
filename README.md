# Airline Complaint Triage

An NLP-based machine learning system that classifies airline customer complaints into six categories and routes them to the appropriate support queue.

## Project Overview

Airline customer complaints can cover different issues such as cancellations, baggage, booking problems, and customer service. The goal of this project is to automatically classify incoming complaint text so that it can be routed to the appropriate support team.

The project compares two classical NLP approaches:

- TF-IDF + Logistic Regression
- TF-IDF + XGBoost

A Streamlit web application is also included for real-time complaint classification.

## Dataset

The project uses the **Twitter US Airline Sentiment** dataset containing 14,640 tweets.

After filtering, cleaning, and removing duplicate records, 7,901 usable negative tweets were used.

The six classification categories are:

- Customer Service
- Delay or Cancellation
- Baggage
- Flight/Crew
- Booking or Fare
- Other

The dataset is not included in this repository.

## Machine Learning Pipeline

```text
Raw Tweets
    ↓
Data Cleaning
    ↓
Duplicate Removal
    ↓
Train / Validation / Test Split
    ↓
Text Preprocessing
    ↓
TF-IDF Feature Extraction
    ↓
Model Training
    ↓
Validation-based Model Selection
    ↓
Final Test Evaluation
    ↓
Streamlit Application
```

The dataset is split into:

- 70% training
- 15% validation
- 15% testing

A fixed random seed is used to make the split reproducible.

## Models

### Logistic Regression

TF-IDF features are combined with Logistic Regression. Hyperparameter tuning is performed using the validation set.

Best validation result:

**Macro-F1: 0.6147**

### XGBoost

TF-IDF features are also used with XGBoost for comparison.

Best validation result:

**Macro-F1: 0.5986**

Based on validation Macro-F1, Logistic Regression was selected as the best classical model.

## Final Test Results

| Model | Macro-F1 | Accuracy | Mean Latency |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.592 | 0.712 | 2.0 ms |
| TF-IDF + XGBoost | 0.591 | 0.715 | 1.7 ms |

The test set contains 1,186 tweets.

The test set was kept separate from model and hyperparameter selection.

## Data Quality Analysis

During preprocessing, a text-label artefact was identified in the source dataset where some label strings had been inserted into tweet text.

The cleaning pipeline repairs the identified patterns before model training.

An ablation experiment was performed to measure the effect of the repair. Logistic Regression achieved:

- Repaired text: Macro-F1 0.592
- Raw text: Macro-F1 0.590

The difference was small on the held-out test set.

## Streamlit Application

The project includes a Streamlit interface where a user can enter an airline complaint and receive a predicted category.

Example:

```text
Input:
"My flight was cancelled and nobody informed me."

Output:
Predicted Category → Delay or Cancellation
```

Run locally with:

```bash
python -m triage.cli app
```

The application runs at:

```text
http://localhost:8501
```

## Project Structure

```text
airline-complaint-triage/
│
├── configs/
├── data/
├── deploy_model/
├── docs/
├── prompts/
├── results/
├── src/
│   └── triage/
│       ├── app.py
│       ├── cli.py
│       ├── data.py
│       ├── text_clean.py
│       ├── evaluate.py
│       ├── error_analysis.py
│       ├── report.py
│       ├── stats.py
│       └── models/
│           ├── classical.py
│           └── llm.py
│
├── tests/
├── .env.example
├── .gitignore
├── ASSUMPTIONS.md
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Installation

Create a Python 3.12 virtual environment:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Place the dataset at:

```text
data/raw/Tweets.csv
```

Prepare the data:

```bash
python -m triage.cli prepare-data
```

Train the classical models:

```bash
python -m triage.cli train-classical
```

Evaluate the models:

```bash
python -m triage.cli evaluate
```

Run the test suite:

```bash
pytest -q
```

Launch the Streamlit application:

```bash
python -m triage.cli app
```

## Technologies

**Languages & Libraries**

Python, Pandas, NumPy, Scikit-learn, XGBoost

**NLP**

TF-IDF, text preprocessing, text classification

**Application**

Streamlit

**Testing**

Pytest

**Tools**

Git, GitHub

## Limitations

- The dataset is from 2015 and contains short, noisy tweets.
- Some classes have considerably fewer examples than others.
- The `other` category is relatively small, making its metrics less stable.
- The routing queues used in the application are illustrative.
- Performance may differ on newer airline data or different customer-service domains.
- The current project is a portfolio/demo system and is not production-ready.

## Future Improvements

- Evaluate the model on a newer airline complaint dataset.
- Experiment with transformer-based text classifiers.
- Improve performance on minority classes.
- Add confidence thresholds and human-review routing.
- Monitor model performance as new complaint data becomes available.
- Experiment with an LLM-based classifier and compare it against the classical models.

## License

The dataset license should be reviewed on the original dataset source before publishing or redistributing the dataset or derived data.