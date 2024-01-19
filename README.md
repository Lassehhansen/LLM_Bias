# World According to LLMs

This project aims to understand the prevalence of disease mentions for 22 different diseases and compare them with GPT-4/Llama 2/Llama 1 Estimates. The goal is to understand biases among disease prevalence according to training data, model outputs, and the real world.

## Introduction

The rise of large language models (LLMs) like GPT-4, fueled by vast textual data and technological advancements, has significantly impacted natural language processing and healthcare. These models, trained on data sources like Common Crawl and Wikipedia, demonstrate advanced text generation and understanding. However, learning from a broad range of web sources renders these models prone to embedding societal biases. Prior work has demonstrated the existence of such biases, including the impact of race on LLM triage decisions and estimates of disease prevalence. The nature and origin of such biases in broad web text data itself, however, remains largely unexplored.

Our research investigates these biases by analyzing web data across various platforms, including Arxiv, Wikipedia, and Common Crawl. We aim to examine how diseases are textually represented in relation to race and gender, uncovering biases in LLM training data.

## Methods
Our analysis involved a detailed co-occurrence study across six critical text corpora: Arxiv, Books, C4, GitHub, StackExchange, and Wikipedia (English), accessed via the RedPajama-Data-1T dataset. This dataset is an open-source implementation of the Llama-1 pre-training environment and influences over 160 open-source LLMs, including those from MosaicML and StabilityAI.

## Data Processing
Text data in JSON Lines format was extracted from these corpora using standard retrieval methods. A specialized filtering pipeline was employed to isolate texts pertinent to our research, focusing on documents containing disease-related terms alongside demographic identifiers.

## Co-Occurrence Analysis
We computed co-occurrence counts for race and gender-related keywords against various diseases within defined token windows. The analysis spanned multiple window sizes to capture immediate and broader contextual associations. This methodology created a co-occurrence matrix, delineating the relationship between diseases and demographic categories. Our primary focus was on the ratio of specific racial or gender mentions relative to total occurrences for each disease, offering a quantifiable perspective on demographic representation in disease-related discourse within these corpora. All code is available at GitHub Repository.

## Repository Structure (Desired):

```
LLM_Bias/
│
├── data/
│   ├── counts/  
│   ├── raw_data/  (Too big to fit here naturally)
│   └── filtered/   (Also too big to fit here)
│
├── src/
│   ├── __init__.py
│   ├── jsonl_data_filtering.py
│   └── data_analysis.py
│
├── dicts/
│   ├── dict_medical.py
│   ├── dict_gender.py
│   └── dict_racial.py
│
├── keywords/
│   ├── keywords_medical.py
│   ├── keywords_gender.py
│   └── keywords_racial.py
│
├── tests/
│   └── test_data_filtering.py
│
├── docs/
│   └── README.md
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── results/
│   └── figures/
│
├── configs/
│   ├── stackexchange.yaml
│   ├── arxiv.yaml
│   ├── wikipedia.yaml
│   ├── c4.yaml
│   ├── books.yaml
│   ├── github.yaml
│   └── commoncrawl.yaml
│
└── main.py
```

## Overview of Text Corpora Analyzed for Bias Analysis in LLMs

| Corpus              | Contents                                                     | Size               |
| --------------------| -------------------------------------------------------------| ------------------ |
| C4                  | A cleaned version of CommonCrawl’s web-crawled corpus        | 175B (807 GB)      |
| GitHub              | Raw GitHub data, filtered by license                         | 59B (213 GB)       |
| Books               | The PG19 subset of the Gutenberg Project and Books3 datasets | 26B (100.4 GB)     |
| arXiv               | Scientific articles from arXiv. Boilerplate removed          | 28B (88 GB)        |
| Wikipedia (English) | All English Wikipedia articles                               | 24B (20.3 GB)      |
| StackExchange       | A network of Q and A websites                                | 20B (74.5 GB)      |

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
```

## How to Use

### Configuration Files
Configuration files are used to specify the parameters for data filtering and analysis. These are stored in the configs/ directory and are written in YAML format. Each dataset should have its own configuration file.

Example of a configuration file (wikipedia.yaml):

```
data:
  input_file_path: 'data/raw_data/wikipedia.jsonl'
  output_folder_path: 'data/filtered/Wikipedia'
  metadata_keys: 
    - "language"
    - "url"
    - "timestamp"
processing:
  remove_latex: true
  save_file: true
  filename: 'wikipedia_filtered.csv'
  total_texts_filename: 'tot_texts_wiki.txt'

```


### Data Filtering and Analysis Scripts

Two main scripts are provided for data processing:

- main_folders.py: For datasets organized in folders with multiple .jsonl files.
- main_single_file.py: For datasets contained in a single .jsonl file.

### Running the Scripts
To run the scripts, use the following command in your terminal, replacing [script_name] with the name of the script you want to run (main_folders or main_single_file) and [config_name] with the name of your configuration file (without the .yaml extension):

```
python [script_name].py [config_name]

```

example:

```
python main_single_file.py wikipedia

```

### This will:

Filter the data based on the terms defined in the dictionaries located in the dicts/ folder.
Perform a co-occurrence analysis.
Save the filtered data and analysis results in specified output directories.

### Data used for visualizations can be found in the visualiations folder.

## Contributing
Contributions to improve the code or add additional functionalities are welcome! Please ensure to follow the existing code structure and comment appropriately.
