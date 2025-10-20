import json
import pandas as pd

# Path to your dataset file
file_path = "data/train_v2.jsonl"

rows = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        text = item["text"]
        acronym = item["acronym"]
        options = item["options"]

        for option_text, is_correct in options.items():
            rows.append({
                "text": text.strip(),
                "acronym": acronym.strip(),
                "option_text": option_text.strip(),
                "label": int(is_correct)
            })

# Create a DataFrame
df = pd.DataFrame(rows)
print(df.head())

df.to_csv("train.csv", index=False)
