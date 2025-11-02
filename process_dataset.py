import argparse
import json
import jsonlines
from datasets import load_dataset

PROMPT_QA = """### Question:
{question}

### Answer:
{answer}"""

PROMPT_Q = """### Question:
{question}

### Answer:"""

def build_datasets_from_hf(hf_name: str, n_samples: int = 1000):
    ds = load_dataset(hf_name, split="train")
    # Try to map common field names
    q_keys = ["question", "instruction", "prompt"]
    a_keys = ["answer", "output", "response"]
    def pick_field(example, keys):
        for k in keys:
            if k in example and example[k] is not None:
                return example[k]
        return None
    examples = []
    for i, ex in enumerate(ds):
        if i >= n_samples:
            break
        q = pick_field(ex, q_keys)
        a = pick_field(ex, a_keys)
        if not q or not a:
            continue
        examples.append({"question": q, "answer": a})
    return examples

def to_jsonl(items, path):
    with jsonlines.open(path, "w") as w:
        w.write_all(items)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_name", type=str, default="lamini/lamini_docs")
    ap.add_argument("--out", type=str, default="data/lamini_docs_processed.jsonl")
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()

    print(f"Loading examples from {args.hf_name} (limit={args.limit})")
    items = build_datasets_from_hf(args.hf_name, args.limit)

    # Apply prompt template to inputs (optional)
    text_only = [{"text": PROMPT_QA.format(question=x["question"], answer=x["answer"])} for x in items]
    qa_split = [{"question": PROMPT_Q.format(question=x["question"]), "answer": x["answer"]} for x in items]

    # Save
    out1 = args.out
    out2 = out1.replace(".jsonl", "_text_only.jsonl")
    to_jsonl(qa_split, out1)
    to_jsonl(text_only, out2)
    print(f"Saved:\\n - {out1}\\n - {out2}")

if __name__ == "__main__":
    main()
