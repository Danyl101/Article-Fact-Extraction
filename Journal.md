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

_____________________________

# Iteration 5 [WEBIE]

The Extracted Data was stored initially as JSON files based on their role(train,val,test) but since those were very heavy on storage decided to zip them using gzip

________________________________________
________________________________________

# Initial Iteration [LLM]

Decided to use a generic LLM model initially (T5-Small) for training initially , to get the data loading and training loops correct , so it was a small model but there were issues with loading the data due to sheer volume of JSON files inside the zip

_________________________________________

# Iteration 1 [LLM]

Decided to instead tokenize the entire dataset beforehand instead of during data loading , which saved alot of time ,but initial tensor files were extremely large (11gb) , which meant accessing even one index required loading the entire file , so instead split it down into smaller chunks making indexing easier

________________________________________

# Iteration 2 [LLM]

Even though the files were significantly smaller, all of them had to be constantly moved into memory ,for proper accessing , which caused high RAM usage , thus decided to use lazy loading , where only the tensor file that needed to be accessed was loaded into the memory

_______________________________________

# Iteration 3 [LLM]

Lazy Loading was still extremely slow , so decided to switch to HF Datasets which had dataloaders that pytorch supported , making them compatible and significantly faster , the json.gz files just had to be converted into .arrow files(HF) only once , and the entire dataloading class was avoided and instead data loading was done directly by DataLoader 

______________________________________

# Iteration 4 [LLM]



