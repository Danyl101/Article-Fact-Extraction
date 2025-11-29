import logging
import traceback
import os
import csv
from config_loader import config

#Checks whether the files have text for each function
def pipeline(text,funcs):
    for func in funcs:
        text=func(text)
        if text is None:
            return None
    return text

#Function to save files
def save_file(filename, full_content,path,suffix=""):
    try:
        if not full_content.strip(): #Checks if there is content
            logging.info(f"No content found for: {filename}")
            return False
    except Exception as e:
        logging.error(f"Content cleaning failed : {e}")
        logging.debug(traceback.format_exc())
        return False

    try:
        if not filename:
            logging.warning(f"Sanitized filename is empty for title: {filename}")
            return False
        name,ext=os.path.splitext(filename)
        filename=f"{name}{suffix}{ext}"if suffix else filename

        os.makedirs(path, exist_ok=True)

        filepath = os.path.join(path, filename) #Creates a file in the necessary folder 
        with open(filepath, "w", encoding="utf-8") as f: #Opens created file 
            f.write(full_content) #Writes content into it

        logging.info(f"Saved: {filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving file {filename} : {e}")
        logging.debug(traceback.format_exc())
        return False
