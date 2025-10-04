# Ragas Semantic Similarity Evaluator

This project provides a simple evaluation pipeline for comparing model outputs against expected outputs using **semantic similarity** with local Hugging Face embeddings.

## Features 🌟

* Compute semantic similarity between model outputs and expected outputs.
* Uses **Ragas** framework and **Hugging Face Embeddings** (`all-MiniLM-L6-v2`) for local inference.
* Save results in a CSV file for easy analysis.
* Normalize text before evaluation to improve accuracy.

## Installation 🚨

1. Clone the repository:

```bash
git clone https://github.com/fuadadhim24/ragas-srl-llm-evaluation.git
cd https://github.com/fuadadhim24/ragas-srl-llm-evaluation.git
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Make sure `sentence-transformers` and `ragas` are installed:

```bash
pip install -U sentence-transformers ragas
```

## Usage 🚀

1. Prepare your dataset in **JSONL** format:

```json
{"prompt": "Show me my tasks in Planning stage", "model_output": "...", "expected_output": "..."}
```

2. Run the evaluation:

```bash
python run_eval.py
```

3. Check the results in:

```
outputs/results.csv
```

The CSV includes columns:

* `prompt`
* `model_output`
* `expected_output`
* `similarity_score` (0–1, higher is better)

## Example

| Prompt                             | Expected Output | Model Output | Similarity Score |
| ---------------------------------- | --------------- | ------------ | ---------------- |
| Show me my tasks in Planning stage | ...             | ...          | 0.542            |

Average, highest, and lowest similarity scores are automatically computed in your analysis.

## Notes 🗑️

* Make sure the dataset file path is correct in `run_eval.py`.
* Text is normalized (lowercased, cleaned of special characters) before evaluation to improve similarity scoring.
* This pipeline is designed for **local evaluation**; no external API calls are needed.
