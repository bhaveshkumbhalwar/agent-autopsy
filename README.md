# 🕵️ Agent Autopsy
### An ML-Powered Failure Diagnosis System for AI Agents

**Agent Autopsy** is a research-oriented project that investigates how machine learning can diagnose failures in AI agents by analyzing their execution traces, tool calls, and error messages.

Instead of simply reporting that an agent failed, the system attempts to classify the failure into a meaningful category to help developers understand what went wrong.

> **Project status:** Research prototype in active development.

## 🎯 Problem Statement

AI agents can fail for several reasons, including unsupported tools, missing arguments, invalid inputs, and execution errors. Understanding these failures is essential for building more reliable AI systems.

Agent Autopsy explores a structured approach to failure diagnosis using execution traces and machine learning.

## ✨ Key Features

- **Standardized execution traces** — Records agent task details, tool calls, execution status, errors, timestamps, and duration.
- **Failure classification** — Uses TF-IDF and Logistic Regression to classify failures.
- **Structured feature extraction** — Extracts indicators such as error presence and recognizable error types.
- **Synthetic dataset generation** — Generates labeled examples for initial model development and testing.
- **Model evaluation** — Evaluates classification performance using accuracy, precision, recall, and F1-score.
- **Generalization experiments** — Tests the baseline model on selected examples from different agent domains.
- **Error analysis** — Records incorrect predictions to help identify weaknesses in the diagnosis process.

## 🧠 Failure Categories

| Category | Description |
|---|---|
| `none` | The tool execution completed successfully. |
| `unknown_tool` | The requested tool is unsupported. |
| `missing_argument` | A required argument is missing. |
| `invalid_input` | The provided input violates expected requirements. |

## ⚙️ How It Works

```text
Agent Task
    ↓
Tool Execution
    ↓
Standardized Execution Trace
    ↓
Feature Extraction
    ↓
ML Failure Classifier
    ↓
Predicted Failure Category
    ↓
Evaluation and Error Analysis
```

The current prototype uses a deterministic tool-execution agent and a supervised machine learning classifier. It is a foundation for researching failure diagnosis across different agent architectures.

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Text Features:** TF-IDF
- **Classification:** Logistic Regression
- **Model Serialization:** Joblib
- **Data Validation:** Pydantic
- **Testing:** Pytest
- **Version Control:** Git and GitHub

## 📊 Current Experimental Results

### Synthetic dataset evaluation

The initial dataset contains 200 synthetically generated examples, with 50 examples per failure category.

| Experiment | Result |
|---|---:|
| Synthetic dataset size | 200 |
| Failure categories | 4 |
| Held-out test examples | 50 |
| Text-only baseline accuracy | 100% |
| Text + structured feature accuracy | 100% |
| Selected cross-agent test accuracy | 66.7% (4/6) |

**Important interpretation:** These are preliminary prototype results, not evidence of real-world diagnostic accuracy. The synthetic examples have relatively consistent error patterns, and the cross-agent evaluation is too small to provide a reliable generalization estimate. Some cross-agent labels also require further review.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/bhaveshkumbhalwar/agent-autopsy.git
cd agent-autopsy
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install the packages required by the project:

```bash
pip install pandas numpy scikit-learn pydantic joblib pytest
```

If additional dependencies are required by your current version, install them from the project's dependency file when available.

### 4. Run the tests

```bash
python -m pytest tests/ -v
```

### 5. Generate the synthetic dataset

```bash
python -m src.evaluation.build_larger_dataset
python -m src.evaluation.prepare_dataset
```

### 6. Train the baseline classifier

```bash
python -m src.diagnosis.train_classifier
```

### 7. Diagnose a trace

Run the prediction module:

```bash
python -m src.diagnosis.predict_failure
```

### 8. Compare models

```bash
python -m src.diagnosis.compare_models
```

Run each command from the project root. The required source files and dependencies must be present.

## 📁 Project Structure

```text
agent-autopsy/
├── src/
│   ├── agents/
│   │   └── basic_agent.py
│   ├── tracing/
│   │   └── trace_schema.py
│   ├── diagnosis/
│   │   ├── features.py
│   │   ├── train_classifier.py
│   │   ├── train_enhanced_classifier.py
│   │   ├── predict_failure.py
│   │   └── compare_models.py
│   └── evaluation/
│       ├── build_dataset.py
│       ├── build_larger_dataset.py
│       ├── prepare_dataset.py
│       ├── test_generalization.py
│       └── analyze_errors.py
├── tests/
├── data/
│   ├── raw/
│   ├── processed/
│   └── traces/
├── models/
├── reports/
├── .gitignore
└── README.md
```

*The tree illustrates the main components developed so far. Check the repository for the exact files present in your current version.*

## 🔬 Research Direction

The long-term goal is to investigate whether failure diagnosis can generalize across different AI agent architectures rather than only recognizing familiar error messages.

Potential next steps include:

- Expanding the dataset with realistic agent execution traces.
- Defining consistent failure labels and improving annotation quality.
- Testing on agent types excluded from training.
- Investigating data leakage and improving evaluation methodology.
- Comparing text-based and structured-feature models.
- Exploring explainable predictions and recovery recommendations.

## ⚠️ Limitations

- Current experiments primarily use synthetic data.
- The current classifier supports four failure categories.
- Cross-agent testing uses a very small sample.
- Reported confidence scores are model estimates and are not necessarily calibrated probabilities.
- The system does not yet autonomously repair failed agents.
- A predicted failure category is a diagnostic suggestion, not proof of the underlying cause.

## 👨‍💻 Author

**Bhavesh Kumbhalwar**

Engineering IT Student | Machine Learning | AI Agent Reliability

GitHub: [@bhaveshkumbhalwar](https://github.com/bhaveshkumbhalwar)

## 📄 License

A license has not yet been specified. Add a `LICENSE` file if you want to define how others may use, modify, and distribute this project.
