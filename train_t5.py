import os
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

def load_jsonl(path: str):
    ds = load_dataset("json", data_files=path, split="train")
    return ds

def preprocess_function(examples, tokenizer, max_source_length, max_target_length):
    inputs = examples["question"]
    targets = examples["answer"]
    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/flan-t5-small")
    parser.add_argument("--train_file", type=str, default="data/sample_lamini_docs_processed.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/flan-t5-small")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--evaluation_strategy", type=str, default="no")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    ds = load_jsonl(args.train_file)

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    print("Tokenizing...")
    tokenized = ds.map(
        lambda batch: preprocess_function(
            batch,
            tokenizer,
            args.max_source_length,
            args.max_target_length,
        ),
        batched=True,
        remove_columns=ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy=args.evaluation_strategy,
        predict_with_generate=True,
        fp16=args.fp16,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    print("Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
