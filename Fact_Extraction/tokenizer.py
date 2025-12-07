import json
import gzip
import torch
from torch.utils.data import Dataset
import logging
from pympler import asizeof

import logging_loader

from transformers import T5Tokenizer, T5ForConditionalGeneration

from config_loader import config

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name,legacy=True)

logger=logging.getLogger("Tokenizer")


def convert_to_str(triples):
    parts = []
    for s, r, o in triples:
        parts.append(f"<subj> {s} <rel> {r} <obj> {o} <et>")
    return " ".join(parts)


class Tokenization:
    def __init__(self, filepath, out_prefix, tokenizer, max_length=512,
                 items_per_chunk=30000):
        self.filepath = filepath
        self.out_prefix = out_prefix
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items_per_chunk = items_per_chunk

        logger.info(f"Starting tokenization for {filepath}")
        self._process()

    def _process(self):
        dataset = []
        chunk_idx = 0

        open_fn = gzip.open if self.filepath.endswith(".gz") else open

        with open_fn(self.filepath, "rt", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                row = json.loads(line)

                input_text = row["input"]
                triples = row.get("mapped_triples", [])

                triple_str = convert_to_str(triples)

                inp_tok = self.tokenizer(
                    input_text,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )

                out_tok = self.tokenizer(
                    triple_str,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )

                dataset.append({
                    "input_ids": inp_tok["input_ids"].squeeze(0),
                    "attention_mask": inp_tok["attention_mask"].squeeze(0),
                    "labels": out_tok["input_ids"].squeeze(0),
                })

                # SAVE CHUNK BY ITEM COUNT
                if len(dataset) >= self.items_per_chunk:
                    save_path = f"{self.out_prefix}_{chunk_idx}.pt"
                    torch.save(dataset, save_path)
                    logger.info(f"Saved {len(dataset)} items → {save_path}")

                    dataset = []
                    chunk_idx += 1

                if line_idx % 2000 == 0:
                    logger.info(f"Processed {line_idx:,} lines")

        # final flush
        if dataset:
            save_path = f"{self.out_prefix}_{chunk_idx}.pt"
            torch.save(dataset, save_path)
            logger.info(f"Saved FINAL chunk → {save_path}")

        logger.info(f"Tokenization COMPLETE for {self.filepath}")
        
train_d=Tokenization(config["paths"]["Dataset"]["Tokenization_Input"]["Train_Data"],"train",tokenizer)
val_d=Tokenization(config["paths"]["Dataset"]["Tokenization_Input"]["Validation_Data"],"val",tokenizer)
test_d=Tokenization(config["paths"]["Dataset"]["Tokenization_Input"]["Test_Data"],"test",tokenizer)
                    
            
        
                
                
