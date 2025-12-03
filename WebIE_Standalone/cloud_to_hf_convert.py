from datasets import Dataset
import json, os
import logging

import logging_loader

from config_loader import config

logger=logging.getLogger("cloud_to_hf_convert")

folder = config["paths"]["WebIE"]["C4_Data_Dir"]

records = []

for root,dirs,files in os.walk(folder):
    for filenames in files:
        try:
            if(filenames.endswith(".json")):
                filepath = os.path.join(root, filenames)
                logger.info(f"Loading file {filepath} with filename {filenames}")
                with open(filepath,encoding="utf-8") as fp:
                    records.append(json.load(fp))
                    logger.debug(f"Loaded file {filenames} with {len(records[-1])} records")
        except Exception as e:
            logger.error(f"Error loading file {filenames}: {e}")
ds = Dataset.from_list(records)
ds.save_to_disk(config["paths"]["WebIE"]["C4_Data_Dir_HF"])
