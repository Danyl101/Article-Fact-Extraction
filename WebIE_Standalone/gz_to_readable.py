import gzip
import json

file_path = "Dataset\\Model_Dataset\\Extracted_Sentences\\train.json.gz"  # your file here

with gzip.open(file_path, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        print(json.dumps(data, indent=2))  # pretty-print
        if i >= 30:  # just show first 5 lines
            break
