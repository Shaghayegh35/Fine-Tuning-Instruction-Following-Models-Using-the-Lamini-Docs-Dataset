# Fine-Tuning Instruction-Following Models (Lamini Docs)

This repository contains a reproducible, minimal pipeline to **process**, **train**, and **evaluate** instruction-following models using the public **Lamini Docs** dataset or your own JSONL files.

## Features
- 🔄 Convert public HF datasets to `question`/`answer` JSONL with prompt templates
- 🧪 Sample dataset included at `data/sample_lamini_docs_processed.jsonl`
- 🧠 Train a small instruction model (`flan-t5-small`) with `scripts/train_t5.py`
- 📝 Jupyter notebook with the end-to-end steps in `notebooks/lamini_finetuning.ipynb`
- 🧰 Clean repo structure (+ `.gitignore`, `LICENSE`, `requirements.txt`)

## Quickstart
```bash
git clone <your-repo-url>
cd lamini-finetuning
python -m venv .venv && source .venv/bin/activate  # or use conda
pip install -r requirements.txt
```

### 1) Prepare data (optional - convert HF dataset)
```bash
python scripts/process_dataset.py --hf_name lamini/lamini_docs --out data/lamini_docs_processed.jsonl --limit 5000
```

### 2) Train (on the sample JSONL or your own)
```bash
python scripts/train_t5.py --model_name google/flan-t5-small --train_file data/sample_lamini_docs_processed.jsonl --output_dir outputs/flan-t5-small --num_train_epochs 1
```

### 3) Evaluate a few predictions
```bash
python scripts/eval.py --model_dir outputs/flan-t5-small --data_file data/sample_lamini_docs_processed.jsonl --n 3
```

> 💡 For larger datasets and faster runs, enable mixed precision: add `--fp16` if your GPU supports it.

## Data Format
Each line in `*.jsonl` should look like:
```json
{"question": "### Question:\nWhat is X?\n\n### Answer:", "answer": "Y..."}
```

## Notes
- This repo is intended as a **minimal**, **portable** starting point. Adjust templates, models, and hyperparameters for your research use case.
- Please ensure you comply with the original dataset licenses when redistributing processed data.

## License
MIT
