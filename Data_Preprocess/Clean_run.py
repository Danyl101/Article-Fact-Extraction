import os
from Clean_functions import is_junk_by_short_lines, regex_cleanup, clean_financial_text, spacy_clean, remove_trailing_noise   
from utils import pipeline, save_file
from config_loader import config
import logging_loader

path=config['paths']['Dataset']['Cleaned_Data_Directory']

def run_clean():
    # Get the path to the BERT_CONTENT folder, relative to the current script
    content_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                  config['paths']['Dataset']['Raw_Data_Directory'])

    # List all files in BERT_CONTENT (filtering to .txt if you want)
    file_list = [f for f in os.listdir(content_folder) if f.endswith('.txt')]

    for filename in file_list:
        file_path = os.path.join(content_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read() #Read text from file 
            text=pipeline(text, [is_junk_by_short_lines, regex_cleanup, clean_financial_text, spacy_clean,remove_trailing_noise]) #Runs pipeline function
            
            if text is not None:
                print(f"Processed {filename}:")
                print(text[:])  # prints entire file 
                print("="*50)
                save_file(filename,text,path) #Saves file
            else:
                print(f"Skipped {filename} due to junk detection or empty content.")

if __name__=="__main__":
    run_clean()