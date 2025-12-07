from datasets import load_dataset
import gzip
import json
import logging
import logging_loader
from config_loader import config

logger=logging.getLogger("Tokenizer")

from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)

def gzip_to_json(filepath,part):
    return load_dataset(
        "json",
        data_files=filepath,
        split=part
    )
    
def has_triples(ex):
    meta = ex.get("meta_obj", {})
    triples = meta.get("mapped_triples", None)
    text = ex.get("input", None)
    return (
        triples is not None and
        isinstance(triples, list) and
        len(triples) > 0 and
        text is not None and
        isinstance(text, str) and
        len(text.strip()) > 0
    )

def convert_triples(example):
    meta = example.get("meta_obj", {})
    triples = meta.get("mapped_triples", None)
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
        max_length=256,
    )
    labels = tokenizer(
        batch["triple_text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )
    # IMPORTANT: Replace tokenizer.pad_token_id (0) with -100 for labels
    labels_ids = labels["input_ids"]
    labels_ids = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
        for seq in labels_ids
    ]      
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__=="__main__":
    dataset=gzip_to_json(config["paths"]["Dataset"]["Tokenization_Input"]["Validation_Data"],"train")
    for i,line in enumerate(dataset):
        if(i<3):
            logger.info(line)
            
    logger.info(f"Raw dataset: {len(dataset)}")
    dataset = dataset.filter(has_triples, num_proc=8)
    
    logger.info(f"After filtering: {len(dataset)}")
    dataset = dataset.map(convert_triples, num_proc=8)
    
    for i,line in enumerate(dataset):
        if(i<3):
            logger.info(line)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        batch_size=1000,
        num_proc=8,      # use all CPU cores
        remove_columns=dataset.column_names,
    )
    
    for i,line in enumerate(tokenized):
        if(i<3):
            logger.info(line)
            
    logger.info(f"After tokenization: {len(tokenized)}")
    
    tokenized.save_to_disk(config["paths"]["Dataset"]["Model_Input"]["Validation_Data"])



