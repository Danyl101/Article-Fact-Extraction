import json
import gzip
import torch
from torch.utils.data import Dataset
import logging 

import logging_loader

from transformers import T5Tokenizer, T5ForConditionalGeneration

from config_loader import config

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name,legacy=True)

logger=logging.getLogger("Tokenizer")

def convert_to_str(triples):
    pieces = []
    for s, r, o in triples:
        pieces.append(f"<subj> {s} <rel> {r} <obj> {o} <et>")
    return " ".join(pieces)


class Tokenization:
    def __init__(self, inp_filepath, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Tokenization started for {inp_filepath}")
        self.data = self._load_and_tokenize(inp_filepath)
        

    def _load_and_tokenize(self, inp_filepath):
        dataset = []
        open_fn = gzip.open if inp_filepath.endswith("gz") else open
        
        try:
            with open_fn(inp_filepath, "rt", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)

                    input_text = row["input"]
                    triples = row.get("mapped_triples", [])

                    triples_str = convert_to_str(triples)

                    inp_tok = self.tokenizer(
                        input_text,
                        max_length=self.max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt"
                    )

                    triples_tok = self.tokenizer(
                        triples_str,
                        max_length=self.max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt"
                    )

                    dataset.append({
                        "input_ids": inp_tok["input_ids"].squeeze(0),
                        "attention_mask": inp_tok["attention_mask"].squeeze(0),
                        "labels": triples_tok["input_ids"].squeeze(0)
                    })
            logger.info(f"Tokenization completed for {inp_filepath}")
            return dataset
        except Exception as e:
            logger.error(f"Tokenization error for {inp_filepath}")
        
train_d=Tokenization(config["paths"]["Dataset"]["Tokenization_Input"]["Train_Data"],tokenizer)
val_d=Tokenization(config["paths"]["Dataset"]["Tokenization_Input"]["Validation_Data"],tokenizer)
test_d=Tokenization(config["paths"]["Dataset"]["Tokenization_Input"]["Test_Data"],tokenizer)

torch.save(train_d,config["paths"]["Dataset"]["Model_Input"]["Train_Data"])
torch.save(val_d,config["paths"]["Dataset"]["Model_Input"]["Validation_Data"])
torch.save(test_d,config["paths"]["Dataset"]["Model_Input"]["Test_Data"])
                    
            
        
                
                
