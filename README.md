# CP-468 Final Project

A Sequence-to-Sequence project for **Grammatical Error Correction (GEC)** comparing:

- a **LSTM encoder-decoder with attention**
- **Google Gemini 3.5 Flash** as the LLM baseline

The goal is to take an incorrect or awkward sentence and generate a corrected version while preserving the original meaning.

---

## Dataset

Training data:
- `agentlans/grammar-correction`
- 10,000 examples

Validation and testing:
- JFLEG
- 755 validation examples
- 748 test examples

The dataset files are stored in:

```text
dataset/
├── train.csv
├── validation.csv
└── test.csv
```

---

## Project Structure

```text
CP-468-Final-Project/
├── data/
│   └── dataset_download.py
├── dataset/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── prompts.py
│   └── llm_baseline.py
├── results/
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Installation

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

---

## LSTM Model

The classical model uses:

- token embeddings
- bidirectional LSTM encoder
- attention mechanism
- LSTM decoder
- teacher forcing
- padding-aware cross-entropy loss
- gradient clipping

### Training

```bash
python src/train.py --epochs 10 --batch-size 32 --embed-size 256 --hidden-size 256 --max-len 50 --output best_model.pt
```

### Evaluation

```bash
python src/evaluate.py --checkpoint best_model.pt --test-csv dataset/test.csv --output lstm_test_predictions.csv
```

### Final LSTM Settings

- Epochs: 10
- Batch size: 32
- Embedding size: 256
- Hidden size: 256
- Maximum sequence length: 50
- Optimizer: Adam
- Learning rate: 0.001
- Random seed: 42
- Teacher forcing ratio: 0.5
- Trainable parameters: 13,039,159

The model was trained using CPU-only PyTorch on an AMD Ryzen 5 7600X.

---

## Gemini LLM Baseline

Google **Gemini 3.5 Flash** is used as the modern LLM comparison model.

Two prompt settings are evaluated:

1. Detailed zero-shot
2. Few-shot with 4 examples

The prompt templates are stored in:

```text
src/prompts.py
```

The LLM evaluation code is stored in:

```text
src/llm_baseline.py
```

---

## Gemini API Setup

Set your Gemini API key as an environment variable.

### PowerShell

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Do not hardcode or commit the API key to GitHub.

---

## Zero-Shot Evaluation

Run the full 748-example zero-shot test:

```bash
python -m src.llm_baseline --variant zero_shot_detailed
```
## Few-Shot Evaluation

Run the full 748-example few-shot test:

```bash
python -m src.llm_baseline --variant few_shot --k 4
```
---

## Output Files

LLM predictions and summary files are saved in:

```text
results/
```

Example files:

```text
results/
├── gemini-3.5-flash_zero_shot_detailed_predictions.csv
├── gemini-3.5-flash_zero_shot_detailed_summary.csv
├── gemini-3.5-flash_few_shot_predictions.csv
└── gemini-3.5-flash_few_shot_summary.csv
```

---

## Reproducibility

Project dependencies are pinned in:

```text
requirements.txt
```

Fixed random seeds are used where appropriate.

API keys and local environment files such as `.env` are excluded through `.gitignore`.

---
