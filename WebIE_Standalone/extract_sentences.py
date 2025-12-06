# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import json
import gzip
import argparse

from tqdm import tqdm
from datasets import load_from_disk
from typing import Dict

import logging

import logging_loader

from config_loader import config

logger= logging.getLogger("extract_sentences")

def load_c4_data(data_dir: str) -> Dict[str, Dict]:
    """
    Load subset c4 data that is used in WebIE.

    Args:
    data_dir (str): The path to filtered c4 articles used in WebIE

    Returns:
    c4_data (Dict): A dictionary storing c4 data {url: {doc, timestamp}}
    """
    c4_data = {}
    ds = load_from_disk(data_dir)
    for i in tqdm(range(len(ds))):
        data = ds[i]
        url = data["url"]
        doc = data["text"]
        timestamp = data["timestamp"]
        c4_data[url] = {"doc": doc, "timestamp": timestamp}
    return c4_data


if __name__ == "__main__":
    """Extract sentence from c4 corresponding to each annotation example."""

    annotation_dir = config["paths"]["Dataset"]["WebIE"]["Annotation_Dir"]
    data_dir = config["paths"]["Dataset"]["Model_Dataset"]["C4_Data_Dir_HF"]
    target_dir = config["paths"]["Dataset"]["Model_Dataset"]["Extracted_Sentences_Dir"]

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f"Load filtered c4 data...")
    c4_data = load_c4_data(data_dir)

    splits = ["train", "val", "test"]

    filenames = {
        "train": ["train_part1.json.gz",
                  "train_part2.json.gz",
                  "train_part3.json.gz",
                  "train_part4.json.gz"],
        "val": ["val.json.gz"],
        "test": ["test.json.gz"],
        "en_test": ["en_test.json.gz"]
    }

    print(f"Extract sentences for each data split...")
    for split in splits:
        print(f"Split: {split}")
        out_f = gzip.open(os.path.join(target_dir, split + ".json.gz"), "wt")
        for fname in filenames[split]:
            ann_file = os.path.join(annotation_dir, fname)
            with gzip.open(ann_file, "rt") as in_f:
                for line in tqdm(in_f):
                    data = json.loads(line)
                    
                    if data["uri"] not in c4_data:
                        logger.error(f"URL not found in c4 data: {data['uri']}")
                        continue
                    else:
                        span = data["span"]
                        url = data["uri"]

                        doc = c4_data[url]["doc"]
                        start, end = span["start"], span["end"]
                        assert start < end

                        sentence = doc[start:end]
                        data["input"] = sentence
                        out_f.write(json.dumps(data) + "\n")
        out_f.close()
    print(f"Complete data is stored at {target_dir}.")
