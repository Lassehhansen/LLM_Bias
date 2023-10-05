import json
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
import re

def create_keyword_pattern(keywords):
    """
    Create a regex pattern for keyword matching.
    
    Parameters:
        keywords (list of str): A list of keywords to create a pattern for.
        
    Returns:
        re.Pattern: A compiled regex pattern that matches any of the provided keywords.
    """
    pattern = r'(?:(?<=\W)|(?<=^))(' + '|'.join(map(re.escape, keywords)) + r')(?=\W|$)'
    return re.compile(pattern, re.IGNORECASE)

def process_chunk_co_occurrence(chunk, medical_patterns, racial_patterns, gender_patterns, window_size):
    """
    Process a chunk of text data to extract co-occurrence of keywords within a specified window size.
    
    Parameters:
        chunk (list of str): A list of text data to be processed.
        medical_patterns (dict): Dictionary of compiled regex patterns for medical keywords.
        racial_patterns (dict): Dictionary of compiled regex patterns for racial keywords.
        gender_patterns (dict): Dictionary of compiled regex patterns for gender keywords.
        window_size (int): The size of the window within which to find co-occurring keywords.
        
    Returns:
        tuple: A tuple containing:
            - co_occurrences_racial (defaultdict): A nested defaultdict containing counts of racial keyword co-occurrences.
            - co_occurrences_gender (defaultdict): A nested defaultdict containing counts of gender keyword co-occurrences.
    """
    co_occurrences_racial = defaultdict(lambda: defaultdict(int))
    co_occurrences_gender = defaultdict(lambda: defaultdict(int))
    
    for line in chunk:
        try:
            entry = json.loads(line)
            text = entry['text']
            for med_key, med_pattern in medical_patterns.items():
                for med_match in med_pattern.finditer(text):
                    start_pos = med_match.start()
                    end_pos = med_match.end()
                    words = text.split()
                    start_word_pos = len(text[:start_pos].split()) - 1
                    end_word_pos = len(text[:end_pos].split())
                    context_words = words[max(0, start_word_pos - window_size):min(len(words), end_word_pos + window_size)]
                    context_str = ' '.join(context_words)
                    
                    for patterns in [racial_patterns, gender_patterns]:
                        for key, pattern in patterns.items():
                            if patterns == racial_patterns:
                                co_occurrences_racial[med_key][key] += len(pattern.findall(context_str))
                            elif patterns == gender_patterns:
                                co_occurrences_gender[med_key][key] += len(pattern.findall(context_str))
        except json.JSONDecodeError as e:
            print(f"Error loading line: {line}. Error: {e}")
    
    return co_occurrences_racial, co_occurrences_gender


def co_occurrence_within_window_parallel(data, medical_dict, racial_dict, gender_dict, window_size):
    """
    Perform parallelized co-occurrence analysis within a window of text.
    
    Parameters:
        data (list of str): A list of text data to be processed.
        medical_dict (dict): Dictionary of medical keywords.
        racial_dict (dict): Dictionary of racial keywords.
        gender_dict (dict): Dictionary of gender keywords.
        window_size (int): The size of the window within which to find co-occurring keywords.
        
    Returns:
        tuple: A tuple containing:
            - df_racial (pd.DataFrame): A DataFrame containing counts of racial keyword co-occurrences.
            - df_gender (pd.DataFrame): A DataFrame containing counts of gender keyword co-occurrences.
    """
    medical_patterns = {k: create_keyword_pattern(v) for k, v in medical_dict.items()}
    racial_patterns = {k: create_keyword_pattern(v) for k, v in racial_dict.items()}
    gender_patterns = {k: create_keyword_pattern(v) for k, v in gender_dict.items()}
    
    num_cores = cpu_count()
    chunk_size = len(data) // num_cores
    
    # Split the data into chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Use multiprocessing Pool to process chunks in parallel
    with Pool(num_cores) as p:
        results = p.map(partial(process_chunk_co_occurrence, 
                                medical_patterns=medical_patterns, 
                                racial_patterns=racial_patterns, 
                                gender_patterns=gender_patterns, 
                                window_size=window_size), 
                        chunks)
    
    # Aggregate results
    co_occurrences_racial = defaultdict(lambda: defaultdict(int))
    co_occurrences_gender = defaultdict(lambda: defaultdict(int))
    
    for racial, gender in results:
        for med_key, counts in racial.items():
            for race_key, count in counts.items():
                co_occurrences_racial[med_key][race_key] += count
        for med_key, counts in gender.items():
            for gender_key, count in counts.items():
                co_occurrences_gender[med_key][gender_key] += count
    
    # Convert to DataFrames
    df_racial = pd.DataFrame(co_occurrences_racial).fillna(0).astype(int).T
    df_gender = pd.DataFrame(co_occurrences_gender).fillna(0).astype(int).T
    
    return df_racial, df_gender


def analyze_data_co_occurrence(source_name, data_path):
    """
    Analyze the data and save various counts and co-occurrence data to CSV files.
    
    Parameters:
        source_name (str): A name for the data source, used in output directory naming.
        data_path (str): Path to the CSV file containing the data to be analyzed.
    """
    df_output = pd.read_csv(data_path)
    
    # Ensure output directory exists
    output_dir = f'output_{source_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1: Total Disease Mention Counts
    total_disease_counts = df_output[list(medical_dict.keys())].sum()
    total_disease_counts.to_csv(os.path.join(output_dir, 'total_disease_counts.csv'))
    
    # 2: Disease Mention Counts Over Different Dates
    disease_date_counts = df_output.groupby('timestamp')[list(medical_dict.keys())].sum().reset_index()
    disease_date_counts.to_csv(os.path.join(output_dir, 'disease_date_counts.csv'))
    
    # 3: Disease Mention Counts with Each Race
    disease_race_counts = df_output.groupby(list(racial_dict.keys()))[list(medical_dict.keys())].sum().reset_index()
    disease_race_counts.to_csv(os.path.join(output_dir, 'disease_race_counts.csv'))
    
    # 4: Disease Mention Counts with Each Gender
    disease_gender_counts = df_output.groupby(list(gender_dict.keys()))[list(medical_dict.keys())].sum().reset_index()
    disease_gender_counts.to_csv(os.path.join(output_dir, 'disease_gender_counts.csv'))
    
    # Co-occurrence within window sizes
    data = df_output['text'].tolist()
    window_sizes = [10, 50, 100, 250]
    
    for window in window_sizes:
        df_racial, df_gender = co_occurrence_within_window_parallel(data, medical_dict, racial_dict, gender_dict, window)
        
        # Ensure sub-directory for window size exists
        window_dir = os.path.join(output_dir, f'window_{window}')
        os.makedirs(window_dir, exist_ok=True)
        
        df_racial.to_csv(os.path.join(window_dir, 'co_occurrence_racial.csv'))
        df_gender.to_csv(os.path.join(window_dir, 'co_occurrence_gender.csv'))
