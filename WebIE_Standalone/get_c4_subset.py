import os
import sys
import json
import argparse
import logging
import io

# Fix path: ROOT = project root (two dirs up from this file)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import logging_loader

from config_loader import config

from datasets import load_dataset, Dataset

logger=logging.getLogger("get_c4_subset")

idx=0  # Starting index for filenames

def save_streamed_item_locally(item: dict, total_urls: int):
    global idx
    # 1. Serialize
    try:
        bytes_data = json.dumps(item, ensure_ascii=False).encode("utf-8")
        logger.debug(f"Serialized item (url={item['url']}, bytes={len(bytes_data)})")
    except Exception as e:
        logger.error(f"Serialization error for URL {item.get('url')}: {e}")
        return

    # 2. Generate filename
    try:
        folder_index = idx // 10000
        folder_path = os.path.join(config["paths"]["WebIE"]["C4_Data_Dir"], str(folder_index))
        os.makedirs(folder_path, exist_ok=True)
        filename=f"{idx}.json"
        filepath=os.path.join(folder_path, filename)
        idx+=1
        logger.debug(f"Saving to: {filepath}")
    except Exception as e:
        logger.error(f"Filename generation error for URL {item.get('url')}: {e}")
        return

    # 3. Write file
    try:
        with open(filepath,"wb") as f:
            f.write(bytes_data)
        logger.info(f"Saved item locally: {filepath}")
    except Exception as e:
        logger.error(f"File write error for URL {item.get('url')}: {e}")
        return
   
if __name__ == "__main__":

    # Paths
    url_path = config["paths"]["WebIE"]["URL_Data"]

    # Load URLs
    try:
        urls = set()
        with open(url_path) as f:
            for line in f:
                urls.add(line.strip())

        print(f"Loaded {len(urls)} urls.")
        logger.info(f"Loaded {len(urls)} urls from {url_path}.")
        logger.debug(f"Sample urls: {list(urls)[:5]}")
    except Exception as e:
        logger.error(f"Error loading urls from {url_path}: {e}")

    print("Filtering c4 for WebIE urls and store in HuggingFace format...")

    urls_set = set(urls)

    # Stream C4
    try:
        stream = load_dataset(
            "allenai/c4",
            "en",
            split="train",
            streaming=True,
        )
        logger.info("Successfully streamed c4 dataset.")
    except Exception as e:
        logger.error(f"Error loading c4 dataset: {e}")
        
    count=0
    try:
        for item in stream:
            if item["url"] in urls_set:
                save_streamed_item_locally(item,len(urls_set))
                count+=1
                logger.debug(f"Matched {count} items so far")
    except Exception as e:
        logger.error(f"Error while filtering dataset: {e}")

