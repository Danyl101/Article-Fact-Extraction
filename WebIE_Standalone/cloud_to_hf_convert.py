from datasets import Dataset
import json, os
import logging

import logging_loader

from config_loader import config

logger=logging.getLogger("cloud_to_hf_convert")

folder = config["paths"]["WebIE"]["C4_Data_Dir_Intermediate"]
files = [os.path.join(folder, f) for f in os.listdir(folder)]

records = []

for f in files:
    try:
        with open(f,encoding="utf-8") as fp:
            records.append(json.load(fp))
            logger.debug(f"Loaded file {f} with {len(records[-1])} records")
    except Exception as e:
        logger.error(f"Error loading file {f}: {e}")

ds = Dataset.from_list(records)
ds.save_to_disk(config["paths"]["WebIE"]["C4_Data_Dir_HF"])
