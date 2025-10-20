import json
from collections import Counter
import pandas as pd

# Path to your file
file_path = "data/train_v2.jsonl"  # <-- Change this to your actual file name

# Load all JSON lines into a list of dictionaries
data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:  # ignore empty lines
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON error: {e}")

# Convert to a DataFrame for convenience
df = pd.DataFrame(data)

# --- Basic stats ---
total_entries = len(df)
unique_acronyms = df['acronym'].nunique()
acronym_counts = df['acronym'].value_counts()

print(f"📊 Total entries: {total_entries}")
print(f"🔠 Unique acronyms: {unique_acronyms}\n")

print("🧾 Frequency of each acronym:")
print(acronym_counts.to_string())

# --- Extract how many true options each acronym has ---
true_option_counts = []
for entry in data:
    acronym = entry["acronym"]
    true_count = sum(1 for val in entry["options"].values() if val)
    true_option_counts.append({"acronym": acronym, "true_options": true_count})

true_df = pd.DataFrame(true_option_counts)

# Group by acronym to get total true options per acronym
summary_df = true_df.groupby("acronym")["true_options"].sum().reset_index()
summary_df = summary_df.sort_values("true_options", ascending=False)

print("\n✅ Number of true options per acronym:")
print(summary_df.to_string(index=False))

# --- Option-level analysis ---
option_true_counts = Counter()
for entry in data:
    for option, val in entry["options"].items():
        if val:
            option_true_counts[option] += 1

option_df = pd.DataFrame(option_true_counts.items(), columns=["Option", "True Count"])
option_df = option_df.sort_values("True Count", ascending=False)

print("\n🏷️ Most frequently true options:")
print(option_df.to_string(index=False))
