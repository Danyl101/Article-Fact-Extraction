
                            VIRTUAL ENVIRONMENT

python -m venv venv

venv\Scripts\activate

py -3.10 -m venv venv 

deactivate 

python -m pip install  
                (If pip is broken in venv)

                                WebIE

get_c4_subset                               

python -m WebIE_Standalone.get_c4_subset 

________________________________________

cloud_to_hf_convert

python -m WebIE_Standalone.cloud_to_hf_convert

_______________________________________

delete_downloaded

python -m WebIE_Standalone.delete_downloaded_c4

______________________________________

extract sentences

python -m WebIE_Standalone.extract_sentences

______________________________________

gz to readable

python -m WebIE_Standalone.gz_to_readable




                                LLM Model


model

python -m Fact_Extraction.model

_______________________________

tokenizer

python -m Fact_Extraction.tokenizer










        