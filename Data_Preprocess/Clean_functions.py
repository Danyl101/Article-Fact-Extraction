import re
import os
from cleantext import clean
import spacy
import logging
import traceback

nlp=spacy.load("en_core_web_sm") #Loading spacy language model

logger=logging.getLogger("Data_Cleaner")

def regex_cleanup(text):
    try:
        logger.info("Starting regex cleanup")
        # Remove author names
        text = re.sub(r'^By\s+[A-Za-z\s]+', '', text, flags=re.MULTILINE)
        
        # Remove timestamps and date lines 
        text = re.sub(r'^\w+\s+\d{1,2},\s+\d{4},?\s*[\d:APMIST\s\(\)]+', '', text, flags=re.MULTILINE)
        
        # Remove time indicators
        text = re.sub(r'\d+\s+Min\s+Read', '', text, flags=re.IGNORECASE)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove repeated multiple newlines or tabs
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\t+', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r' +', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
    except Exception as e:
        logger.error("Error occured during regex cleaning") #Log message
        logger.error(traceback.format_exc())
    return text

def clean_financial_text(text):#Normalize with clean-text 
    try:
        logger.info("Starting financial text cleaning")
        text = clean(text,
                    fix_unicode=True,
                    to_ascii=True,
                    lower=False,           # keep casing for FinBERT
                    no_line_breaks=True,   # remove line breaks for continuous text
                    no_urls=True,
                    no_emails=True,
                    no_phone_numbers=True,
                    no_digits=False,       # keep digits (prices, dates)
                    no_currency_symbols=False)  # keep currency symbols
    except Exception as e:
        logger.error("Error occurred during financial text cleaning") #Log message
        logger.error(traceback.format_exc())
    return text

def spacy_clean(text):#SpaCy Cleaning
    try:
        doc=nlp(text) #SpaCy nlp extracts info from text
        sentences=(sent.text.strip() for sent in doc.sents if len(sent.text.strip())>20) #Filter out short sentences
        cleaned_text="".join(sentences) #Joins filtered sentences
    except Exception as e:
        logger.error("Error occurred during SpaCy cleaning") #Log message
        logger.error(traceback.format_exc())
    return cleaned_text

def is_junk_by_short_lines(text, max_words=20, threshold=4): #Checks for junk based on sequence of short lines
    try:
        lines = text.split('\n') #Splits line on newline character
        consecutive_count=0
        max_count=0
        for line in lines:
            if(len(line)<max_words): #Check if line is short 
                consecutive_count+=1
                if(max_count<consecutive_count):
                    max_count=consecutive_count
            else:
                consecutive_count=0
    except Exception as e:
        logger.error("Error during junk file cleaning")
        logger.error(traceback.format_exc())
    if(max_count <= threshold): #Checks if sequence of short lines are below threshold
        return text
    else:
        return None

    
def remove_trailing_noise(text,window=200,trim_chars=20): #Removes Trailing Spaces
    noise_phrases=["continue reading",
            "read more",
            "subscribe",
            "advertisement",
            "follow us on",
            "click here",
            "read next"
            "all rights reserved",]
    text_lower=text.lower()
    text_snippet=text_lower[-window:] #Takes the last portion of text
    snippet_start=len(text_lower)-window #Defines start of snippet
    for phrase in noise_phrases:
        try:
            idx = text_snippet.find(phrase) #Finds index of phrase
            abs_idx=snippet_start+idx #Finds absolute index from whole text
            logger.info("Phrase found")
            if abs_idx != -1:
                print(abs_idx)

                # Calculate slice boundaries
                start = max(abs_idx - trim_chars, 0) #Finds the start of slicing
                end = min(abs_idx + len(phrase) + trim_chars, len(text)) #Finds the end of slicing
                logger.info("Start and end of slicing identified")

                # Remove snippet around the noise phrase
                new_text = text[:start] + text[end:] #Slices the full text
                logger.info("Slicing done ")
                return new_text  # remove just the first found phrase and return
        except Exception as e:
            logger.error("Indexing failed during trailing")
            logger.error(traceback.format_exc())





    
    

    
