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

                self.data.append({
                    "input": input_text,
                    "label": triple_str
                })

    def convert_triples(self, triples):
        # Format: <subj> S <rel> R <obj> O <et>
        pieces = []
        for s, r, o in triples:
            pieces.append(f"<subj> {s} <rel> {r} <obj> {o} <et>")
        return " ".join(pieces)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        inputs = self.tokenizer(
            row["input"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        labels = self.tokenizer(
            row["label"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # reshape to remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = labels["input_ids"].squeeze(0)

        # Replace pad token id with -100 for loss masking
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels
        }
