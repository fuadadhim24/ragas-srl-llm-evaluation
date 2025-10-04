import json

with open("data/dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        print(entry["prompt"], len(entry["model_output"]))
