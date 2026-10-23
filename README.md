
# Agent Autopsy
### An ML-Powered Failure Diagnosis System for AI Agents

**Agent Autopsy** is a research-oriented project that investigates how machine learning can diagnose failures in AI agents by analyzing their execution traces, tool calls, and error messages.

Instead of simply reporting that an agent failed, the system attempts to classify the failure into a meaningful category to help developers understand what went wrong.

> **Project status:** Research prototype in active development.

## Problem Statement.

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

## Tech Stack

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
=======
# Agent Autopsy

A failure-diagnosis prototype for AI agent execution traces.

Agent Autopsy takes a trace of a tool-calling agent, decides which of four
failure categories it belongs to, and reports the evidence behind the
decision. The repository also contains the evaluation harness built to
find out how much that classifier actually knows: a leakage audit, a
grouped split, an independent test set, a baseline/combined comparison,
a cross-agent test set, a labelling-rule audit and a small HTTP API.

> **Status:** research prototype. Not production ready. The numbers below
> are measured on small, hand-authored and synthetic datasets, and the
> README says exactly what they do and do not support.

## What it does

```text
agent task ──> tool execution ──> standardised trace
                                        │
                                        ▼
                        text features (+ optional structured features)
                                        │
                                        ▼
                        TF-IDF -> LogisticRegression
                                        │
                                        ▼
                     predicted failure category + class scores
```

Four categories are supported, and nothing else is:

| Category | Meaning |
|---|---|
| `none` | The tool call completed successfully. |
| `unknown_tool` | The requested tool is not in the agent's registry. |
| `missing_argument` | A required argument was absent from the call. |
| `invalid_input` | A supplied argument had an unusable type or value. |

A failure that happened *after* a valid call is **not** forced into one of
these. Generated code that raises `NameError`, a 404 from a real tool and
a permission error are all outside the taxonomy, and the code reports
them as a coverage gap instead of inventing a label. The rules are in
[`src/evaluation/labelling_rules.py`](src/evaluation/labelling_rules.py).

## Architecture

| Module | Responsibility |
|---|---|
| `src/tracing/` | `AgentTrace` schema, label vocabulary, JSONL read/write. |
| `src/agents/basic_agent.py` | The deterministic calculator agent that produces training traces. |
| `src/evaluation/` | Dataset builders, leakage audit, splitting, metrics, model and cross-agent evaluation, label audit. |
| `src/diagnosis/` | Feature definitions, pipelines, training, inference, model comparison. |
| `src/api/app.py` | FastAPI service: `/health` and `/diagnose`. |
| `tests/` | 209 tests covering the trace contract, datasets, splits, leakage, metrics, models, API and reporting. |

The feature definitions live in one place (`src/diagnosis/features.py`), and
training and inference both build their input through the same scikit-learn
`Pipeline`, so the two cannot drift apart. Task ids, timestamps and
durations are never model inputs.

## Datasets

Four datasets exist, and they are not interchangeable.

| Dataset | Size | How it was produced | Used for |
|---|---|---|---|
| Training data (`data/processed/ml_dataset.csv`) | 200 rows, 50 per label | **Synthetic agent executions**: the calculator agent is really run on seeded random tasks | Fitting both models |
| In-domain evaluation (`data/evaluation/in_domain_eval.csv`) | 48 rows, 12 per label | **Hand-authored simulation**, never executed | The final in-domain test set |
| Cross-agent evaluation (`data/evaluation/cross_agent_eval.csv`) | 48 rows (33 labelled, 15 out-of-taxonomy) across calculator / Python / web / file | **Hand-authored simulation**, never executed | Testing the taxonomy and the model on other agent domains |
| Generalisation suite (`src/evaluation/test_generalization.py`) | 6 cases (4 labelled, 2 out-of-taxonomy) | Hand-written examples | A small contract suite plus the label audit |

The training traces are *executed* but their tasks are synthetic. The
evaluation traces are *simulated*: they were written by hand and no agent
was run to produce them. **The repository contains no real captured
production traces.** Every evaluation row carries a `provenance` field
saying so, and the reports repeat it.

The training data is generated by:

```bash
python -m src.evaluation.build_larger_dataset   # runs the agent, writes traces + failure_dataset.csv
python -m src.evaluation.prepare_dataset        # validates labels, writes ml_dataset.csv
```

`build_larger_dataset` seeds Python's `random` with 42, so the tasks are
reproducible. One column is not: `duration_ms` is real wall-clock time and
changes between runs. It is excluded from the model input, so every score
in this README reproduces exactly.

The evaluation sets are regenerated deterministically and never read by a
training script:

```bash
python -m src.evaluation.build_eval_dataset
python -m src.evaluation.build_cross_agent_dataset
```

## Models

Two models are trained, and they differ in exactly one flag:

```python
Pipeline([
    ("features", TraceFeatureExtractor(use_structured_features=False)),  # text model
    ("classifier", LogisticRegression(...)),
])
```

| Model | File | Input |
|---|---|---|
| `text` | `models/failure_classifier.joblib` | TF-IDF of `agent_type`, `tool_name`, `status`, `error_type`, `error` |
| `combined` | `models/combined_failure_classifier.joblib` | The same TF-IDF block plus the five structured features |

The structured features are: `has_error`, `has_arguments`,
`has_missing_argument_error`, `has_unknown_tool_error`,
`has_invalid_input_error`. They are heuristic phrase detectors. The text
block is identical in both models, and a test asserts that the only
difference between the two pipelines is the flag.

Train them with:

```bash
python -m src.diagnosis.train_classifier
python -m src.diagnosis.train_combined_model
```

`src/diagnosis/train_enhanced_classifier.py` is a deprecated alias for
`train_classifier`; it no longer prints a leaked-split score.

## Splitting and leakage

The first version of this project reported 100% accuracy. It was wrong.
The 200 training rows collapse into **16 distinct model inputs** behind
**4 template families** (184 rows are duplicates in the model's eyes). A
plain stratified random split therefore puts a literal copy of every test
row into the training set:

| Split | Test rows copied from training | Accuracy |
|---|---:|---:|
| Stratified random (old) | 50/50 (100%) | 1.0000 |
| Grouped (current) | 0/105 (0%) | see below |

The leakage audit also probes identifier shortcuts: `task_id` predicts
the label for 98.5% of ordinal neighbours while the text-only model alone
scores 24%, so the id encodes the label and is excluded. `duration_ms` is
near chance (neighbour probe 24.1%), which is consistent with it being
noise, and it is also excluded.

The current split is a deterministic, stratified **group** assignment:
rows sharing an exact model input stay in the same fold, and each fold has
all four labels. There are **two folds**, because three of the four label
families have only two template groups; with more folds some label would
be missing from some test set. Results are pooled out-of-fold
predictions over all 200 rows.

## Results

All numbers below were produced by the commands in this README, from the
committed code. Reports are in `reports/`.

### Grouped cross-validation (200 rows)

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| text | 1.0000 | 1.0000 |
| combined | 1.0000 | 1.0000 |

This score is **not evidence of generalisation**: within a fold, the four
templates are trivially separable. It is reported for completeness, and
the number that matters is the independent set below.

### Independent in-domain test set (48 hand-authored cases)

| Model | Accuracy | Macro F1 | 95% bootstrap CI (accuracy) |
|---|---:|---:|---|
| text | 0.8125 | 0.8150 | [0.688, 0.917] |
| combined | 0.8542 | 0.8516 | [0.750, 0.958] |

The combined model is 2 cases better out of 48 (delta +0.0417 accuracy).
Six predictions changed: two cases moved from wrong to right
(`eval-031`, `eval-032`, both "Missing required argument" reported as
`ValueError`) and four moved from one wrong label to another
(`eval-039/040/045/046`, truly `invalid_input`, predicted
`missing_argument` instead of `unknown_tool`). An exact McNemar test on
the two discordant cases gives **p = 0.500**. The improvement is not
statistically supportable, and it is a side effect of re-weighting, not a
principled gain.

Per class, independent set:

| Class | text P/R/F1 | combined P/R/F1 | support |
|---|---|---|---|
| `invalid_input` | 1.000 / 0.583 / 0.737 | 1.000 / 0.583 / 0.737 | 12 |
| `missing_argument` | 0.833 / 0.833 / 0.833 | 0.667 / 1.000 / 0.800 | 12 |
| `none` | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 12 |
| `unknown_tool` | 0.588 / 0.833 / 0.690 | 0.909 / 0.833 / 0.870 | 12 |

By stress kind (text → combined):

| Stress | text | combined |
|---|---|---|
| seen template | 6/6 | 6/6 |
| unseen tool | 14/15 | 14/15 |
| unseen error type | 10/10 | 10/10 |
| **paraphrased message** | **9/17 (0.529)** | **11/17 (0.647)** |

Paraphrase is where the model fails. Five of the 48 cases share an exact
model input with training; the rest do not.

### Cross-agent test set (48 hand-authored cases, 33 labelled)

| Domain | Correct / labelled |
|---|---:|
| calculator (the training domain) | 8/9 |
| python execution | 6/8 |
| web | 5/8 |
| file operations | 6/8 |
| **total** | **25/33 = 0.7576** |

Macro F1 is 0.7609, and the 95% bootstrap interval over 33 cases is
**[0.606, 0.879]**. By case kind: in-domain transfer 6/6, new agent
13/15, paraphrased 6/12.

The coverage measurement is the important part:

| Coverage | Rows |
|---|---:|
| exact model input seen in training | 4/48 |
| `agent_type` seen in training | 12/48 (only `calculator`) |
| `tool_name` seen in training | 11/48 |
| error message seen in training | 12/48 |

Three of the four agent types never appear in training. All eight
misclassifications were predicted `unknown_tool`: when the model sees an
agent type, tool or wording it has never met, `unknown_tool` is its
default answer. That is a novelty bias, not cross-agent understanding.
Thirteen of the fifteen out-of-taxonomy traces were also forced onto
`unknown_tool`; they are counted as coverage gaps, not errors.

**This does not show broad generalisation.** It shows what a
calculator-only model does when it is asked about a domain it was never
trained on.

### Generalisation label audit

The six original generalisation cases were re-audited against the
labelling rules. Every declared label agrees with the rules. The old
exception-type mapping would have mislabelled three of the six:

| Case | Exception | Old mapping | Rules |
|---|---|---|---|
| `gen-001` | `ValueError` | `invalid_input` | `unknown_tool` |
| `gen-005` | `NameError` | `invalid_input` | out of taxonomy |
| `gen-006` | `HTTPError` | `unknown_tool` | out of taxonomy |

`gen-005` and `gen-006` are the two errors the audit was written to catch:
a `NameError` in generated code is not a bad argument, and a 404 from a
registered tool is not an unknown tool.

### What the structured features really do

The combined model is only worth keeping if the structured features carry
information the text does not. Measured on the training data:

* `has_arguments` fires on 200/200 rows and its learned coefficient is
  exactly **0.0000** for every class: it is constant and useless.
* `has_missing_argument_error`, `has_unknown_tool_error` and
  `has_invalid_input_error` each fire on exactly 50/200 rows and, on
  those rows, exactly one label is ever present. They are copies of the
  label, not general signals.
* Their learned weights are +1.55 to +1.61 on their own class and about
  −0.51 to −0.54 elsewhere, which is why `has_error` still carries weight
  (+0.50 to +0.53, −1.57 for `none`).

That is why the combined model's two-case gain must not be read as
"structured features work". They are exact-match phrase detectors; a
paraphrase fires nothing.

## Running it

### 1. Environment

Python **3.9** is required (the code is checked on 3.9.6 and avoids 3.10
syntax such as `str | None`).

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is a full, pinned freeze, including FastAPI and
Uvicorn. The test suite and every command below run in a fresh venv
created from it.

### 2. Build the data and the models

```bash
python -m src.evaluation.build_larger_dataset
python -m src.evaluation.prepare_dataset
python -m src.evaluation.build_eval_dataset
python -m src.evaluation.build_cross_agent_dataset

python -m src.diagnosis.train_classifier
python -m src.diagnosis.train_combined_model
```

### 3. Evaluate

```bash
python -m src.evaluation.evaluate_model              # leakage audit + splits + independent set
python -m src.diagnosis.compare_models               # text vs combined
python -m src.evaluation.evaluate_cross_agent        # four agent domains
python -m src.evaluation.test_generalization         # 6-case contract suite
python -m src.evaluation.analyze_errors              # misclassifications from the suite
python -m src.evaluation.audit_generalization_labels # label-rule audit
```

Reports are written to `reports/`:

| Report | Contents |
|---|---|
| `evaluation_metrics.json` | Leakage audit, split summaries, independent-set metrics, predictions |
| `evaluation_confusion_matrix*.csv` | Confusion matrices |
| `evaluation_predictions*.csv` | Per-row predictions |
| `evaluation_by_case_kind.csv` | Accuracy per stress kind |
| `leakage_audit.json` | Duplicate groups, split overlap, identifier probes |
| `model_comparison.json` | Both models, all splits, deltas, CIs, McNemar, feature diagnostics |
| `model_comparison_*.csv` | Predictions, confusion matrices, per-kind accuracy |
| `model_comparison_structured_features.json` | Per-feature firing rates and label distributions |
| `cross_agent_eval.json` | Cross-agent metrics, coverage, uncertainty, limitations |
| `cross_agent_predictions.csv` | Every cross-agent row with prediction and novelty flags |
| `cross_agent_error_analysis.csv` | Misclassifications with the novelty around each |
| `cross_agent_coverage.csv` | Per-row train/eval coverage |
| `generalization_results.csv` | The 6-case suite, scored vs out-of-taxonomy |
| `error_analysis.csv` | Misclassifications from that suite |
| `generalization_label_audit.csv` | Exception-type mapping vs rules vs declared label |

### 4. Diagnose one trace

```bash
python -m src.diagnosis.predict_failure --model text
python -m src.diagnosis.predict_failure --model combined
```

### 5. Run the API

```bash
uvicorn src.api.app:app --reload
```

Open <http://127.0.0.1:8000/docs> for the interactive schema, or use curl:

```bash
curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/diagnose \
  -H 'content-type: application/json' \
  -d '{
    "task_id": "demo-1",
    "agent_type": "calculator",
    "tool_name": "divide",
    "status": "failure",
    "error_type": "ValueError",
    "error": "Unknown tool: divide"
  }'
```

```json
{
  "task_id": "demo-1",
  "predicted_label": "unknown_tool",
  "class_scores": {
    "invalid_input": 0.0375,
    "missing_argument": 0.0385,
    "none": 0.0364,
    "unknown_tool": 0.8877
  },
  "model": "text"
}
```

`class_scores` are the classifier's scores, **not calibrated
probabilities**. The model was never calibrated, so neither the API nor
this README calls them probabilities or confidence.

| Status | Meaning |
|---|---|
| 200 | Diagnosis returned. |
| 422 | The request failed validation (missing `task_id`, bad `status`, non-object body). |
| 500 | Inference failed. The trace and internals are logged, not returned. |
| 503 | No model is loaded; the detail says which command trains it. |

The model is loaded **once** during application startup and cached on
`app.state`; a request never reads the disk. If the model file is
missing, the service still starts and `/health` reports
`"model_loaded": false` rather than crashing. There is no
authentication, rate limiting or TLS: this is a local service, not a
deployment.

## Tests

```bash
python -m pytest -q
```

**209 tests pass**, 0 fail, 0 skip when the generated datasets exist (four
dataset-consistency tests skip if you have not run the builders). They
cover:

* the trace schema, status/label consistency and JSONL round-tripping;
* dataset builders, label validity and the refusal to guess a label;
* leakage audit, grouped splitting, determinism and metric definitions;
* both pipelines: identical text block, feature order, row-order
  independence, missing fields, unknown tools, missing arguments, invalid
  inputs, successful traces, unseen error messages, target-leak guard;
* model save/load and prediction on a new trace;
* the comparison statistics (bootstrap interval, exact McNemar);
* the labelling rules and the cross-agent scoring workflow;
* the API: health, validation, 500/503 handling, one load per process,
  and the wording of the class-score documentation.

## Limitations and known issues

1. **Small, simulated data.** Training is 200 synthetic executions; the
   independent set is 48 hand-authored rows; the cross-agent set is 48.
   Intervals are wide and are reported with every accuracy.
2. **No real traces.** There is no captured production agent behaviour
   anywhere in the repository.
3. **Template collapse.** 200 training rows are 16 distinct model inputs
   from 4 templates. Grouped cross-validation scores 1.0 because the
   templates are separable, not because the model generalises.
4. **Paraphrase weakness.** Accuracy on rephrased error messages is 0.529
   for the baseline and 0.647 for the combined model on 17 cases.
5. **Novelty bias.** An unknown agent type, tool or wording is usually
   answered `unknown_tool`; all 8 cross-agent errors and 13 of the 15
   out-of-taxonomy predictions did this.
6. **The combined model's gain is not significant.** Two net cases out of
   48, McNemar p = 0.500, with four other rows merely changing from one
   wrong label to another.
7. **Structured features are label copies in training.**
   `has_arguments` is constant (200/200, coefficient 0.0000) and the
   three phrase detectors fire for exactly one label each.
8. **Four categories only.** Execution failures (runtime errors, HTTP
   failures, filesystem errors, timeouts) have no label and cannot be
   scored. They are reported as coverage gaps.
9. **`duration_ms` is wall-clock**, so `data/processed/ml_dataset.csv`
   differs in that column between runs. It is excluded from the model
   input, so reported scores reproduce.
10. **The trace log grows.** `build_larger_dataset` appends to
    `data/traces/standardized_agent_traces.jsonl`, so that tracked file
    changes every time the pipeline runs.
11. **No authentication, TLS or rate limiting** on the API.
12. **No license** has been chosen for this repository.

## Project structure

```text
agent-autopsy/
├── src/
│   ├── agents/basic_agent.py          # deterministic calculator agent
│   ├── tracing/                       # AgentTrace, labels, JSONL I/O
│   ├── diagnosis/
│   │   ├── features.py                # single source of feature definitions
│   │   ├── modeling.py                # pipelines, model registry
│   │   ├── training.py                # shared training routine
│   │   ├── train_classifier.py        # text-only baseline
│   │   ├── train_combined_model.py    # text + structured
│   │   ├── predict_failure.py         # load + diagnose
│   │   └── compare_models.py          # controlled comparison
│   ├── evaluation/
│   │   ├── build_larger_dataset.py    # synthetic training data
│   │   ├── prepare_dataset.py         # validation -> ml_dataset.csv
│   │   ├── build_eval_dataset.py      # in-domain test set
│   │   ├── build_cross_agent_dataset.py
│   │   ├── labelling_rules.py         # immediate cause vs underlying cause
│   │   ├── leakage.py, splitting.py   # leakage audit and grouped splits
│   │   ├── metrics.py, runner.py      # metrics, CIs, shared runners
│   │   ├── evaluate_model.py
│   │   ├── evaluate_cross_agent.py
│   │   ├── test_generalization.py
│   │   ├── analyze_errors.py
│   │   └── audit_generalization_labels.py
│   └── api/app.py                     # FastAPI service
├── tests/                             # 209 tests
├── data/
│   ├── traces/                        # append-only agent trace log (tracked)
│   ├── raw/, processed/               # generated, ignored
│   └── evaluation/                    # generated evaluation sets (small)
├── models/                            # generated joblib files, ignored
├── reports/                           # generated evidence, quoted above
├── requirements.txt                   # full pinned freeze (Python 3.9)
└── README.md
```

## Author

**Bhavesh Kumbhalwar**

## License

No license has been specified. Add a `LICENSE` file before distributing
this project.
>>>>>>> eec0e8e (updated the project, evaluation, and other)
