# Buggy Code Fine-Tuning Project

This repository contains a Jupyter Notebook designed for fine-tuning a Large Language Model (specifically `Qwen/Qwen2.5-Coder-3B-Instruct`) to identify and fix bugs in code. It was built to run in a Kaggle environment.

## Features
* **Supervised Fine-Tuning (SFT):** Uses `trl` and `peft` (QLoRA) to fine-tune the model in 4-bit or 8-bit precision.
* **Batch Training:** Supports chunked dataset loading so large datasets can be trained across multiple Kaggle sessions without losing progress.
* **Evaluation via GPT:** Uses the Cerebras API (via OpenAI Python client) to automatically score the model's generated code fixes before and after fine-tuning.
* **Direct Preference Optimization (DPO):** Includes a pipeline to generate preference pairs (using different sampling temperatures) and fine-tune the model further to align with better code generation.

## Dependencies
* `transformers`
* `peft`
* `trl`
* `bitsandbytes`
* `accelerate`
* `datasets`
* `sentencepiece`
* `openai`

## Usage
1. Upload the notebook to a Kaggle environment with GPU access.
2. Update the `DATABASE_PATH` to point to your SQLite database containing the `buggy_code` and `fixed_code` pairs.
3. Configure your API keys (e.g., `OPENROUTER_API_KEY`) for the evaluation steps.
4. Run the notebook sequentially.
