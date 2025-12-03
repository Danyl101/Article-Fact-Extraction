                    POST CONSTRUCTION LOG 


# TAGS=[WEBIE,LLM]


# Initial Iteration[WEBIE]

Decided to use WebIE by amazon to extract data , its a program that extracts textual data from c4 dataset , and then associates these files with annotations which were manually annotated , the annotations comes in the two major forms , span&output , which essentially extracts the most important portion in the article , which is irrelevant for us , and extracts fact in form of triplets which we needed 

____________________________

# Iteration 1 [WEBIE]

So the webie repo used c4 data and loaded them into disk , which was extremely storage intensive and required around 500gb of storage , so decided to directly upload the data into google drive and then access it as needed , also c4 dataset had deteriorated and we needed to load the entire dataset (800gb) from which we then collected our needed data around 2gb , which was very inefficient , so decided to use allenai/c4 which supported streaming , allowing us to extract only needed data

____________________________

# Iteration 2 [WEBIE]

So drive api was taking extremely long time to upload files due to throttling limits and since we only needed to download 2gb of data now instead of 800gb , decided to download it directly into local storage , chunking each text file into a json file and storing it , minor tweaks into storage were done , like instead of storing all files into one directly instead splitting into different sub directories once 10k files was reached

_____________________________

# Iteration 3 [WEBIE]

Extracted all the json files needed , but the extraction portion or essentially matching files to their annotations required them to be in huggingface format (.arrow) , so converted all the json files into arrow files

_____________________________

# Iteration 4 [WEBIE]

Annotation extraction was done now , but with minor changes implemented , due to the difference in c4 dataset itself(old c4 vs new allenai) there were issues with timestamp formatting , which caused a validation step to be skipped , the main matching came in the form of matching the url present in every json file with the url file given in annotation files , since urls wont change



