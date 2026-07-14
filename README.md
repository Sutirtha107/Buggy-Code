# Automated Bug Fixing with Qwen 2.5 Coder

This repository contains a comprehensive pipeline for fine-tuning a Large Language Model (`Qwen/Qwen2.5-Coder-3B-Instruct`) to automatically identify and fix bugs in source code across multiple programming languages (C, C++, JavaScript, Java, Ruby, Python, PHP, Go, and C#).

The pipeline is designed to run in a Kaggle environment and features Supervised Fine-Tuning (SFT), automated LLM-as-a-Judge evaluation, and Direct Preference Optimization (DPO).

## 🚀 Technical Approach

### 1. Data Preparation & Chunked Training
Because bug-fix datasets can be massive, the notebook implements a **Chunked Training System**:
* Data is loaded from an SQLite database (`runbugrun.db`).
* The data is deterministically split (Train/Val/Test) using a fixed seed to prevent data leakage across sessions.
* Extremely long outliers (>1500 tokens) are filtered out via batch tokenization to prevent out-of-memory (OOM) errors and slow processing.
* The notebook trains on manageable `CHUNK_SIZE` batches (e.g., 2000 samples) per Kaggle session. It saves the LoRA adapter weights and resumes from them in the next session (`RESUME_FROM_CHUNK`), circumventing Kaggle's maximum session time limits.

### 2. Supervised Fine-Tuning (SFT)
* **Quantization:** Uses `bitsandbytes` NF4 (4-bit) quantization to fit the model into consumer GPU memory.
* **QLoRA:** Utilizes `peft` (Parameter-Efficient Fine-Tuning) to target specific projection layers (`q_proj`, `k_proj`, `v_proj`, etc.) without altering the base model weights.
* **Early Stopping:** Training automatically halts if the validation loss doesn't improve for a set number of evaluations, preventing overfitting and saving compute time.

### 3. Automated Evaluation (LLM-as-a-Judge)
Instead of manual testing, the pipeline automatically evaluates the model's performance **before** and **after** fine-tuning:
* It holds out a strict test set that the model has never seen.
* It generates fixes for these buggy snippets.
* It uses an external API (Cerebras via the OpenAI client interface) to score the generated fixes from 0 to 10 based on Correctness, Logic, and Code Quality.
* It outputs a direct `comparison.json` showing the absolute point improvement gained from the fine-tuning.

### 4. Direct Preference Optimization (DPO)
To further align the model's output with high-quality code generation, the notebook includes a DPO phase:
* It generates two fixes for the same bug using different sampling temperatures (a deterministic low-temperature fix and a creative high-temperature fix).
* These pairs are scored by the LLM-Judge.
* The model is trained using the `DPOTrainer` to push its probability distribution toward the higher-scoring (preferred) fixes and away from the lower-scoring (rejected) fixes.

## 📦 Dependencies
* `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate` (Hugging Face stack)
* `datasets`, `sentencepiece`
* `openai` (For the Cerebras evaluation API)

## 🛠️ Usage
1. Upload this notebook to a Kaggle environment with GPU access (e.g., T4 x2 or P100).
2. Attach your SQLite database containing the `bugs` and `problems` tables.
3. Update `DATABASE_PATH` in the notebook to point to your dataset.
4. Set your API key (`OPENROUTER_API_KEY`) for the evaluation phase.
5. Run the notebook. For subsequent runs on the same dataset, increment `RESUME_FROM_CHUNK` to continue training where the previous session left off.
