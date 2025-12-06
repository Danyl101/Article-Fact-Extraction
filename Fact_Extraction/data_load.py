import json
import gzip
from torch.utils.data import Dataset

class TripleExtractionDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load file (supports .gz or normal jsonl)
        self.data = []
        open_fn = gzip.open if filepath.endswith(".gz") else open

        with open_fn(filepath, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

                input_text = row["input"]
                triples = row.get("mapped_triples", [])

                # Convert triples list → linear string
                triple_str = self.convert_triples(triples)
                
                tok_inp = self.tokenizer(
                    input_text,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )

                tok_lbl = self.tokenizer(
                    triple_str,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )

                self.data.append({
                    "input_ids": tok_inp["input_ids"].squeeze(0),
                    "attention_mask": tok_inp["attention_mask"].squeeze(0),
                    "labels": tok_lbl["input_ids"].squeeze(0)
                })
                
             # Replace pad tokens with -100 in labels
        pad_id = self.tokenizer.pad_token_id
        for row in self.data:
            row["labels"][row["labels"] == pad_id] = -100

    def convert_triples(self, triples):
        # Format: <subj> S <rel> R <obj> O <et>
        pieces = []
        for s, r, o in triples:
            pieces.append(f"<subj> {s} <rel> {r} <obj> {o} <et>")
        return " ".join(pieces)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
