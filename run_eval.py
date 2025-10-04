import os
import json
import csv
import asyncio
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import SemanticSimilarity
from ragas.embeddings import HuggingFaceEmbeddings

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "results.csv")
DATASET_FILE = "data/dataset.jsonl" 

hf_embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
semantic_scorer = SemanticSimilarity(embeddings=hf_embeddings)

def normalize(text):
    if not text:
        return ""
    text = text.lower()
    text = text.replace("*", "").replace("–", "-").replace("—", "-")
    return " ".join(text.split())

dataset = []
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))

results = []

async def compute_similarity(model_output, expected_output):
    sample = SingleTurnSample(
        response=normalize(model_output),
        reference=normalize(expected_output)
    )
    score = await semantic_scorer.single_turn_ascore(sample)
    return score

for entry in dataset:
    prompt = entry.get("prompt", "")
    model_output = entry.get("model_output", "")
    expected_output = entry.get("expected_output", "")

    similarity_score = 0.0
    if expected_output and model_output:
        similarity_score = asyncio.run(compute_similarity(model_output, expected_output))

    results.append({
        "prompt": prompt,
        "model_output": model_output,
        "expected_output": expected_output,
        "similarity_score": round(similarity_score, 4)
    })

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["prompt", "model_output", "expected_output", "similarity_score"]
    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(results)

print(f"[Evaluator] Finished! Results saved to {OUTPUT_FILE}")
