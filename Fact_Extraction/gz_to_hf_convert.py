from datasets import load_dataset
import gzip
import json

from config_loader import config

from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)

def gzip_to_json(filepath,part):
    load_dataset(
        "json",
        data_files=filepath,
        split=part
    )

def convert_triples(example,data):
    triples = example.get("mapped_triples", [])
    text = []

    for s, r, o in triples:
        text.append(f"<subj> {s} <rel> {r} <obj> {o} <et>")

    example["triple_text"] = " ".join(text)
    return example

def tokenize(batch):
    model_inputs = tokenizer(
        batch["input"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    labels = tokenizer(
        batch["triple_text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__=="__main__":
    dataset=gzip_to_json(config["paths"]["Dataset"]["Tokenization_Input"]["Train_Data"],"Train")
    dataset = dataset.map(convert_triples, num_proc=8)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        batch_size=1000,
        num_proc=8,      # use all CPU cores
        remove_columns=dataset.column_names,
    )
    tokenized.save_to_disk(config["paths"]["Dataset"]["Model_Input"]["Train_Data"])



