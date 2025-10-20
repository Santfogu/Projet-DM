from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import json

model_path = "./results"  # path to your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

test_file = "data/test_v4.jsonl"

all_preds = []
all_labels = []

with open(test_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        text = item["text"].strip()
        acronym = item["acronym"].strip()
        options_raw = item["options"]  # might be dict or list

        # Normalize options to a list of (option_text, label) pairs
        normalized_options = []
        if isinstance(options_raw, dict):
            for opt_text, is_correct in options_raw.items():
                normalized_options.append((opt_text.strip(), int(is_correct)))
        elif isinstance(options_raw, list):
            for opt in options_raw:
                if isinstance(opt, dict):
                    opt_text = opt.get("option_text") or opt.get("text") or opt.get("option") or ""
                    label = opt.get("label")
                    if label is None:
                        label = opt.get("is_correct") or opt.get("correct") or opt.get("true")
                    # try to infer a numeric/bool label if still None
                    if label is None:
                        for v in opt.values():
                            if isinstance(v, (int, bool)):
                                label = int(bool(v))
                                break
                    normalized_options.append((str(opt_text).strip(), int(bool(label)) if label is not None else 0))
                else:
                    # list of plain strings
                    normalized_options.append((str(opt).strip(), 0))
        else:
            raise ValueError(f"Unsupported options format: {type(options_raw)}")

        # For this item, find the predicted option with highest probability
        scores = []
        option_texts = []
        option_labels = []
        for opt_text, is_correct in normalized_options:
            input_text = f"{text} {acronym} : {opt_text}"
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=256
            )
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                score = probs[0][1].item()
                scores.append(score)
                option_texts.append(opt_text)
                option_labels.append(int(is_correct))

        best_idx = int(torch.tensor(scores).argmax())
        predicted_option = option_texts[best_idx]

        # Collect predictions and true labels per option
        for idx, true_label in enumerate(option_labels):
            pred_label = 1 if idx == best_idx else 0
            all_preds.append(pred_label)
            all_labels.append(true_label)

# Now compute metrics on all options over entire dataset
metrics = compute_metrics(all_preds, all_labels)
print(metrics)
