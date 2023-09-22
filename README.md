# World According to LLMs

This project aims to understand the prevalence of disease mentions for 19 different diseases and compare them with GPT-4/Llama 2/Llama 1 Estimates. The goal is to understand biases among disease prevalence according to training data, model outputs, and the real world.

## Research Questions:

- How do skews in race-disease associations in pre-training data relate to race and gender biases in model outputs?
- What metrics can we use to predict healthcare-related biases in outputs?
- How well is “the world as it is” (true disease distributions) reflected in these open-source corpora?
- Measuring co-occurrence between disease terms and race/ethnicity and gender keywords in pre-training data.
- Can these results inform us about bias in the final model outputs? I.e., do co-occurrence ratios in pre-training data correlate with fine-tuned LLM bias?
- How well do the pre-training datasets reflect true disease-demographic relationships? I.e., how well does the distribution of disease-demographic associations in the training data match ‘true’ distributions?

## Extracting Relevant Text from Training Data:

### Step 1:
Define keyword dictionaries that relate to each disease, race, and gender.

### Step 2:
Filter all documents from pre-training data that mention disease AND (gender OR race).

### Step 3:
Deal with ambiguous keywords, e.g., ensuring all mentions of ‘white’ and ‘black’ relate to ethnicity and not actual colors (”the black car was for sale”), and that ‘aids’ relate to the disease and not the unrelated noun (hearing or walking aids) and the verb (she ‘aids’ him). This is done using a biomedical NER tagger that is configured to only extract keyword matches that are classified as pertaining to the disease or the race (personal background). In this step, all irrelevant occurrences of the keywords are flagged, so they don’t count in the subsequent co-occurrence analysis.

## Datasets

Note the datasets analyzed are all English datasets. 

- Arxiv (88 GB)
    - Total data loaded and filtered (Keyword Present - Medical AND Racial OR Gender): 77788
    - Filtered Size (Keyword Present - Medical AND Racial OR Gender): 4.6 GB
    - Filtered Size (Ambgious Keywords Filtering): 
- GitHub (213 GB)
    - Filtered Size (Keyword Present - Medical AND Racial OR Gender): 2.8 GB
    - Filtered Size (Ambgious Keywords Filtering): 
- Stackexchange (74.5 GB)
    - Filtered Size (Keyword Present - Medical AND Racial OR Gender): XX GB
    - Filtered Size (Ambgious Keywords Filtering): 
- Wikipedia (112 GB) -> When filtered for English only (20.3GB)
    - Filtered Size (Keyword Present - Medical AND Racial OR Gender): 1.6 GB
    - Filtered Size (Ambgious Keywords Filtering): 
- Commoncrawl
    - 2022-05 Folder: (251 GB)
    - 2023-06 Folder: (289 GB)
    - Filtered Size (Keyword Present - Medical AND Racial OR Gender): XX GB
- C4 (807 GB)
    - Filtered Size (Keyword Present - Medical AND Racial OR Gender): 19.4 GB
    - Total data loaded and filtered: 2340188
- Books (100.4 GB)
    - Filtered Size (Keyword Present - Medical AND Racial OR Gender): 52.3 GB

## Downloading the Training Data:

To download only the files pertaining to RedPyjama from the dataset, follow the steps below. Detailed documentation for the download can be found [here](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T).

```bash
# Download the urls.txt file which contains URLs to all the datasets
wget 'https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt'

# Get the urls related to each of the datasets in RedPajama
grep “arxiv” urls.txt > arxiv_urls.txt
# ... [repeat for other datasets]

# Use the modified script to download only the specific files
while read line; do
    dload_loc=${line#https://data.together.xyz/redpajama-data-1T/v1.0.0/}
    mkdir -p $(dirname $dload_loc)
    wget "$line" -O "$dload_loc"
done < arxiv_urls.txt
# ... [repeat for other datasets]


# for Wikipedia, select only english articles:
jq -c 'select(.meta.language == "en")' wiki.jsonl > wiki_en.jsonl

# if you dont have jq:

## on Ubuntu:
sudo apt-get install jq

## on macOS with Homebrew:
brew install jq
