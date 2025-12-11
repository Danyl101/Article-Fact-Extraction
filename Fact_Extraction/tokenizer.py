from datasets import load_dataset
import gzip
import json
import logging
import logging_loader
from config_loader import config

logger=logging.getLogger("Tokenizer")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large") #Defines tokenizer to be used

def gzip_to_json(filepath,part): #HuggingFace functions loads the .json.gz files
    return load_dataset(
        "json",
        data_files=filepath,
        split=part
    ) 
    
def has_triples(ex): #Filter to check if inputs or triples are empty 
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
    
def is_grounded(subject, obj, text):
    """
    Returns True if BOTH subject and object appear in the input text.
    Case-insensitive exact substring match.
    """
    text_low = text.lower()
    return (subject.lower() in text_low) and (obj.lower() in text_low)

def filter_and_convert_triples(example):
    meta = example.get("meta_obj", {})
    triples = meta.get("mapped_triples", [])
    inp = example.get("input", "")

    clean = []
    seen = set()

    for (s, r, o) in triples:
        key = (s.lower(), r.lower(), o.lower())

        # remove duplicates
        if key in seen:
            continue
        seen.add(key)

        # NEW: keep only grounded triples
        if is_grounded(s, o, inp):
            clean.append((s, r, o))

    # If no triples left → drop this example
    if len(clean) == 0:
        example["triple_text"] = ""   # Needed temporarily; will be filtered later
        return example

    # Convert to REBEL-ish text format (tab separated)
    out = []
    for (s, r, o) in clean:
        out.append(
        f"<triplet> <subj> {s} </subj> "
        f"<rel> {r} </rel> "
        f"<obj> {o} </obj> </triplet>"
    )

    example["triple_text"] = " ".join(out)   # Recommended by REBEL
    return example

def tokenize(batch): #Tokenizes the input strings and label strings
    model_inputs = tokenizer(
        batch["input"],
        padding=False,
        truncation=True,
        max_length=256,
    )
    labels = tokenizer(
        batch["triple_text"],
        padding=False,
        truncation=True,
        max_length=64,
    )
    # Replace tokenizer.pad_token_id (0) with -100 for labels
    labels_ids = labels["input_ids"]
    labels_ids = [
        [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
        for seq in labels_ids
    ]
    
    model_inputs["labels"] = labels_ids
    return model_inputs

if __name__=="__main__":
    dataset=gzip_to_json(config["paths"]["Dataset"]["Tokenization_Input"]["Train_Data"],"train")
    dataset = dataset.train_test_split(test_size=0.05, seed=42)["test"]
    for i,line in enumerate(dataset):
        if(i<3):
            logger.info(line)
            
    logger.info(f"Raw dataset: {len(dataset)}")
    dataset = dataset.filter(has_triples, num_proc=8)
    
    logger.info(f"After filtering: {len(dataset)}")
    dataset = dataset.map(filter_and_convert_triples, num_proc=8)
    
    for i,line in enumerate(dataset):
        if(i<3):
            logger.info(line)

    tokenized = dataset.map(
        tokenize,
        batched=True,
        batch_size=1000,
        num_proc=4,      # use half CPU cores
        remove_columns=dataset.column_names,
    )
    
    for i,line in enumerate(tokenized):
        if(i<3):
            logger.info(line)
            
    logger.info(f"After tokenization: {len(tokenized)}")
    
    tokenized.save_to_disk(config["paths"]["Dataset"]["Model_Input"]["Train_Data"])



