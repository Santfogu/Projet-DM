from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import json
import sys
import traceback

# === 1. Load and expand your JSON lines dataset ===
file_path = "data/train_v2.jsonl"  # <-- put your dataset filename here

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

# Optional: check what it looks like
df = pd.DataFrame(rows)
print(df.head())

# === 2. Convert into a Hugging Face Dataset ===
dataset = Dataset.from_pandas(df)

# === 3. Tokenize ===
tokenizer = AutoTokenizer.from_pretrained("camembert-base")  # French-friendly model

def preprocess(example):
    # map(..., batched=True) provides lists — build a single input string per item
    texts = example["text"]
    acronyms = example["acronym"]
    options = example["option_text"]

    if not isinstance(texts, list):
        texts = [texts]
        acronyms = [acronyms]
        options = [options]

    inputs = [t.strip() + " " + a.strip() + " : " + o.strip()
              for t, a, o in zip(texts, acronyms, options)]

    tokenized = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=256
    )

    # keep labels under key 'labels' for Trainer
    tokenized["labels"] = example["label"]
    return tokenized

dataset = dataset.map(preprocess, batched=True)

# Split into train/validation sets (e.g., 90/10)
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# === 4. Initialize model ===
model = AutoModelForSequenceClassification.from_pretrained("camembert-base", num_labels=2)

# === 5. Training configuration ===
# Some transformer versions don't accept newer kwargs (e.g., evaluation_strategy).
# Try the modern constructor first; if it fails (TypeError), fall back to a compatible set.
try:
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01
    )
except TypeError as e:
    # Likely an older transformers version where evaluation_strategy is not supported.
    print("Warning: TrainingArguments raised TypeError when using 'evaluation_strategy'.")
    print("Falling back to older-compatible arguments (omitting evaluation_strategy).")
    # Optionally show the original error for debugging
    traceback.print_exception(e, e, e.__traceback__, file=sys.stdout)
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        do_eval=True  # older flag that may be recognized
    )


# === 6. Trainer setup ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# === 7. Train ===
trainer.train()
