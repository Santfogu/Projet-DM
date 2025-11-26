# Acronym Association Project

This project focuses on identifying the correct expansions of railway-related French acronyms in context using various machine learning approaches.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Open any notebook in Jupyter or VS Code and run the cells sequentially. Note:
   - You'll need a CUDA-compatible GPU for the LLM inference
   - The notebooks use 4-bit quantization to reduce memory requirements
   - You will need at least 40GB of available disk space to run the Mistral-Small model, and around 60Gb for QWEN-32B



## Notebooks Overview

### 1. `naive-solution.ipynb`
Basic approach that queries a Large Language Model (LLM) directly without using training data. Tests different models (Mistral-7B) to establish a baseline performance for acronym expansion without any examples and with very basic ones.

### 2. `embedding_based_examples.ipynb`
Improves upon the naive solution by using semantic embeddings to find relevant training examples. Uses the `multilingual-e5-base` model to:
- Compute embeddings for both training and test data
- Find the top-k most similar examples for each test case based on cosine similarity
- Provide these examples to the LLM for better context-aware predictions

### 3. `local-validation.ipynb`
Validates the embedding-based approach using a train/test split (80/20) on the training data. This allows for:
- Local testing and hyperparameter tuning (especially the k parameter for number of examples)
- Performance evaluation before running on the actual test set
- Experimentation with different prompting strategies including "Train of Thought" (ToT) approach

### 4. `tot-k4.ipynb`
Implements the final approach using k=4 examples with a Train of Thought prompting strategy. Uses Qwen or Mistral models to analyze each option systematically before providing the final answer. Generates predictions for the test set.

### 5. `tot-k4-camembert.ipynb`
Similar to `tot-k4.ipynb` but uses pre-computed CamemBERT embeddings (loaded from pickle files) for finding relevant examples. This notebook produces the final predictions saved to CSV format for submission.

## Data
- `train_v2.jsonl`: Training data with acronyms, texts, and correct expansions
- `test_v4.jsonl`: Test data for predictions

## Predictions
The `predictions/` folder contains various CSV files with model outputs from different approaches and configurations.
