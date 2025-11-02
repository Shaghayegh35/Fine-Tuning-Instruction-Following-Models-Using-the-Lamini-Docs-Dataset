import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="outputs/flan-t5-small")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--data_file", type=str, default="data/sample_lamini_docs_processed.jsonl")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)

    ds = load_dataset("json", data_files=args.data_file, split="train")
    for i in range(min(args.n, len(ds))):
        q = ds[i]["question"]
        a = ds[i]["answer"]
        inputs = tokenizer(q, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=128)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("="*80)
        print(q)
        print("\nExpected:", a)
        print("\nPredicted:", pred)

if __name__ == "__main__":
    main()
