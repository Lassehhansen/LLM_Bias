import pandas as pd
import re
import json
import os 
from multiprocessing import Pool, cpu_count
from keywords import medical_keywords, racial_keywords, gender_keywords
from functools import partial
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm  


######################################## Initialize the NER pipeline #######################################################
tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")
pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")

############################################################################################################################

def create_keyword_pattern(keywords):
    """Helper function to create a compiled regex pattern from a list of keywords.
    
    Args:
        keywords (list of str): List of keywords to be matched.

    Returns:
        pattern: A compiled regex pattern.
    """
    pattern = r'(?:(?<=\W)|(?<=^))(' + '|'.join(map(re.escape, keywords)) + r')(?=\W|$)'
    return re.compile(pattern, re.IGNORECASE)

def keyword_present(text, medical_keyword_pattern, racial_keyword_pattern, gender_keyword_pattern):
    """Check if a medical keyword AND either a racial OR gender keyword are present in the given text.
    
    Args:
        text (str): The text to be checked.
        medical_keyword_pattern (re.Pattern): Compiled regex pattern for medical keywords.
        racial_keyword_pattern (re.Pattern): Compiled regex pattern for racial keywords.
        gender_keyword_pattern (re.Pattern): Compiled regex pattern for gender keywords.

    Returns:
        bool: True if the text contains the specified keywords, otherwise False.
    """
    return bool(medical_keyword_pattern.search(text)) and (bool(racial_keyword_pattern.search(text)) or bool(gender_keyword_pattern.search(text)))

def remove_latex_commands(s):
    """Remove common LaTeX commands from a string.
    
    Args:
        s (str): The string containing potential LaTeX commands.

    Returns:
        str: The string with LaTeX commands removed.
    """
    # Replace escape sequences and actual newline/tab characters with space
    s = re.sub(r'\\[nrt]|[\n\r\t]', ' ', s)
    # Remove LaTeX commands
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    # Remove other potential LaTeX delimiters or unwanted characters
    s = re.sub(r'\\.', '', s)  # Removes escaped characters like \\, \{, \}, etc.
    # Remove LaTeX environments (this is a basic version and might not catch all environments)
    s = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '', s, flags=re.DOTALL)
    # Remove math mode content
    s = re.sub(r'\$.*?\$', '', s)
    s = re.sub(r'\\[.*?\\]', '', s)
    s = re.sub(r'\\\(.*?\\\)', '', s)
    s = re.sub(r'\\\[.*?\\\]', '', s)
    # Remove delimiters before and after words
    s = re.sub(r'(?<=\W)\\|\\(?=\W)', '', s)
    return s.strip()

def save_to_json(data, output_folder_path, filename="filtered_data.json"):
    """Save the data to a .json file.
    
    Args:
        data (list): List of data entries.
        output_folder_path (str): Path to the output folder.
        filename (str, optional): Name of the output file. Defaults to "filtered_data.json".
    """
    output_path = os.path.join(output_folder_path, filename)
    
    # Ensure the directory exists. If not, create it.
    os.makedirs(output_folder_path, exist_ok=True)
    
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

def process_file(file_path, medical_keyword_pattern, racial_keyword_pattern, gender_keyword_pattern, remove_latex=True):
    """Process and filter lines in a given file."""
    combined_data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            try:
                entry_data = json.loads(line)
                text_data = entry_data['text']
                if remove_latex:
                    text_data = remove_latex_commands(text_data)
                lowered_text = text_data.lower()
                if keyword_present(lowered_text, medical_keyword_pattern, racial_keyword_pattern, gender_keyword_pattern):
                    combined_data.append({
                        'text': lowered_text,
                        'meta_data': entry_data.get('meta', {})
                    })
            except json.JSONDecodeError as e:
                print(f"Error loading line in {file_path}: {line}. Error: {e}")
    return combined_data

def process_and_filter_json_files(input_folder_path, output_folder_path, medical_keywords, racial_keywords, gender_keywords, remove_latex=True, save_file=True, filename="filtered_data.json"):
    """Process, filter, and optionally save data from .jsonl files."""

    medical_keyword_pattern = create_keyword_pattern(medical_keywords)
    racial_keyword_pattern = create_keyword_pattern(racial_keywords)
    gender_keyword_pattern = create_keyword_pattern(gender_keywords)

    # Changed this line to collect file paths instead of os.DirEntry objects
    file_paths = [entry.path for entry in os.scandir(input_folder_path) if entry.is_file() and entry.name.endswith('.jsonl')]

    combined_data = []

    # Use multiprocessing Pool to process files in parallel
    with Pool(cpu_count()) as p:
        results = p.map(partial(process_file, 
                                medical_keyword_pattern=medical_keyword_pattern, 
                                racial_keyword_pattern=racial_keyword_pattern, 
                                gender_keyword_pattern=gender_keyword_pattern,
                                remove_latex=remove_latex), 
                        file_paths)
        
    for result in results:
        combined_data.extend(result)

    if save_file:
        save_to_json(combined_data, output_folder_path, filename)

    print(f"Total data loaded and filtered: {len(combined_data)}")
    return combined_data

def process_chunk(chunk, medical_keywords, racial_keywords, gender_keywords, remove_latex=True):
    # Compile the regex patterns within the function
    medical_keyword_pattern = create_keyword_pattern(medical_keywords)
    racial_keyword_pattern = create_keyword_pattern(racial_keywords)
    gender_keyword_pattern = create_keyword_pattern(gender_keywords)
    
    combined_data = []
    for line in chunk:
        try:
            entry_data = json.loads(line)
            text_data = entry_data['text']
            if remove_latex:
                text_data = remove_latex_commands(text_data)
            lowered_text = text_data.lower()
            if keyword_present(lowered_text, medical_keyword_pattern, racial_keyword_pattern, gender_keyword_pattern):
                combined_data.append({
                    'text': lowered_text,
                    'meta_data': entry_data.get('meta', {})
                })
        except json.JSONDecodeError as e:
            print(f"Error loading line: {line}. Error: {e}")
    return combined_data

def process_large_file_in_parallel(file_path, medical_keywords, racial_keywords, gender_keywords, output_folder_path, remove_latex=True, save_file=True, filename="filtered_data.json"):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Determine the number of available CPU cores
    num_cores = cpu_count()

    # Calculate the chunk size based on the number of lines and cores
    chunk_size = len(lines) // num_cores

    # If there's a remainder, add an extra chunk for the remaining lines
    if len(lines) % num_cores != 0:
        num_cores += 1

    # Split the lines into chunks
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    combined_data = []

    # Use multiprocessing Pool to process chunks in parallel
    with Pool(num_cores) as p:
        results = p.map(partial(process_chunk, 
                                medical_keywords=medical_keywords, 
                                racial_keywords=racial_keywords, 
                                gender_keywords=gender_keywords,
                                remove_latex=remove_latex), 
                        chunks)
        
    for result in results:
        combined_data.extend(result)

    # Save the combined data if the save_file flag is True
    if save_file:
        save_to_json(combined_data, output_folder_path, filename)

    print(f"Total data loaded and filtered: {len(combined_data)}")
    return combined_data

def modify_text_for_ambiguous_keywords(text, medical_pattern, racial_pattern, window_size=10):
    replacements = {}
    words = text.split()

    for pattern, keyword_type in [(medical_pattern, 'medical'), (racial_pattern, 'racial')]:
        for match in pattern.finditer(text):
            word_index = len(text[:match.start()].split())
            start_idx = max(0, word_index - window_size)
            end_idx = min(len(words), word_index + window_size)
            context_words = words[start_idx:end_idx]
            context = ' '.join(context_words)

            entities = pipe(context)
            word_to_entity = {entity['word'].lower(): entity['entity_group'] for entity in entities}

            if keyword_type == 'medical' and word_to_entity.get(match.group().lower()) in ["Disease_disorder", "History"]:
                continue
            elif keyword_type == 'medical':
                replacements[match.span()] = "medical_missclassification"
            
            if keyword_type == 'racial' and word_to_entity.get(match.group().lower()) == "Personal_background":
                continue
            elif keyword_type == 'racial':
                replacements[match.span()] = "racial_missclassification"

    for span, replacement in reversed(sorted(replacements.items())):
        start, end = span
        text = text[:start] + replacement + text[end:]
    
    return text

def process_texts(chunk, ambiguous_medical_keywords, ambiguous_racial_keywords, meta_data_cols, window_size=10, start_index=0):
    results = []
    
    # Create patterns for the ambiguous keywords using create_keyword_pattern
    medical_pattern = create_keyword_pattern(ambiguous_medical_keywords)
    racial_pattern = create_keyword_pattern(ambiguous_racial_keywords)
    
    # Wrap the data with tqdm for progress bar
    for idx, entry in tqdm(enumerate(chunk, start=start_index), total=len(chunk), desc="Processing texts"):
        old_text = entry['text']
        new_text = modify_text_for_ambiguous_keywords(old_text, medical_pattern, racial_pattern, window_size)
        
        medical_replacements = new_text.count("medical_missclassification")
        racial_replacements = new_text.count("racial_missclassification")
        
        meta_data_values = [entry['meta_data'][col] for col in meta_data_cols]
        
        result = {
            'clean_text': new_text,
            'text_index': idx,
            'no_ambigous_medical_keywords_replaced': medical_replacements,
            'no_racial_keywords_replaced': racial_replacements,
            'length_old_data': len(old_text),
            'length_new_data': len(new_text)
        }
        for col, value in zip(meta_data_cols, meta_data_values):
            result[col] = value
        
        results.append(result)

    return pd.DataFrame(results)

def parallel_process_texts(data, ambiguous_medical_keywords, ambiguous_racial_keywords, meta_data_cols, window_size=10):
    num_cores = cpu_count()
    chunk_size = len(data) // num_cores

    # If there's a remainder, add an extra chunk for the remaining lines
    if len(data) % num_cores != 0:
        num_cores += 1

    # Split the data into chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    # Use multiprocessing Pool to process chunks in parallel
    with Pool(num_cores) as p:
        results = p.starmap(process_texts, [(chunks[i], ambiguous_medical_keywords, ambiguous_racial_keywords, meta_data_cols, window_size, i*chunk_size) for i in range(num_cores)])

    # Combine the results
    combined_df = pd.concat(results, ignore_index=True)

    return combined_df





