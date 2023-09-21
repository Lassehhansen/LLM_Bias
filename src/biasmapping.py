import re
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from keywords import medical_keywords, racial_keywords, gender_keywords
from dicts import medical_keywords_dict, racial_keywords_dict, gender_keywords_dict
from corpusprocessing import create_keyword_pattern

def getkeywords():
    """
    Fetches and returns the lists of keywords.
    
    Returns:
    dict: A dictionary containing:
        - medical_keywords: List of medical-related keywords from keywords.py
        - racial_keywords: List of racial-related keywords from keywords.py
        - gender_keywords: List of gender-related keywords from keywords.py
    """
    return {
        'medical_keywords': medical_keywords,
        'racial_keywords': racial_keywords,
        'gender_keywords': gender_keywords
    }

def getdicts():
    """
    Fetches and returns the keyword dictionaries for different categories.
    
    Returns:
    dict: A dictionary containing:
        - medical_keywords_dict: Dictionary for medical-related keywords from dicts.py
        - racial_keywords_dict: Dictionary for racial-related keywords from dicts.py
        - gender_keywords_dict: Dictionary for gender-related keywords from dicts.py
    """
    return {
        'medical_keywords_dict': medical_keywords_dict,
        'racial_keywords_dict': racial_keywords_dict,
        'gender_keywords_dict': gender_keywords_dict
    }

def document_cooccurrence(df, medical_keywords, racial_keywords, medical_keywords_dict, racial_keywords_dict):
    """
    Calculates document-level co-occurrence between medical and racial or gender keywords in a given dataframe.
    This is calculated for invididual keyword matches in the sub-categories.

    Parameters:
    - input_df (pd.DataFrame): The input dataframe with a 'text' column containing the texts for analysis.
    - medical_keywords (list of str): List of medical-related keywords.
    - racial_keywords (list of str): List of race-related keywords.
    - medical_keywords_dict (dict): Dictionary mapping broad medical categories to specific keywords.
    - racial_keywords_dict (dict): Dictionary mapping broad racial categories to specific keywords.

    Returns:
    - co_occurrence_df (pd.DataFrame): A dataframe showing the co-occurrence counts between medical and racial keywords.
    - aggregated_df (pd.DataFrame): An aggregated dataframe showing the co-occurrence counts between broad categories of medical and racial terms.

    Example usage:
    >>> co_occurrence_df, aggregated_df = document_cooccurrence(df, medical_keywords, racial_keywords, medical_keywords_dict, racial_keywords_dict)
    """
    # 1. Pre-compile regex patterns for all keywords
    def generate_pattern(keyword):
        return r'(?:^|[\s.,;!?])' + re.escape(keyword) + r'(?:$|[\s.,;!?])'
    
    med_patterns = {keyword: re.compile(generate_pattern(keyword), re.IGNORECASE) for keyword in medical_keywords}
    racial_patterns = {keyword: re.compile(generate_pattern(keyword), re.IGNORECASE) for keyword in racial_keywords}

    # 2. Define function to find all matching keywords for a text and pattern set
    def find_matches(text, patterns):
        return {keyword for keyword, pattern in patterns.items() if pattern.search(text)}

    # 3. Create the co-occurrence matrix
    co_occurrence_matrix = [[0 for _ in range(len(racial_keywords))] for _ in range(len(medical_keywords))]

    # For each text, find matching keywords and update the matrix
    for text in df['text']:
        matched_med_keywords = find_matches(text, med_patterns)
        matched_racial_keywords = find_matches(text, racial_patterns)
        
        for med_keyword in matched_med_keywords:
            for racial_keyword in matched_racial_keywords:
                i = medical_keywords.index(med_keyword)
                j = racial_keywords.index(racial_keyword)
                co_occurrence_matrix[i][j] += 1

    # 4. Convert the matrix to a DataFrame
    df_co_occurrence = pd.DataFrame(co_occurrence_matrix, columns=racial_keywords, index=medical_keywords)
    df_co_occurrence['medical_keyword'] = df_co_occurrence.index
    df_co_occurrence = df_co_occurrence.melt(id_vars=["medical_keyword"], var_name="racial_keyword", value_name="co_occurrence_count")
    
    # 5. Aggregation logic
    disease_mapping = {subcat.lower(): cat for cat, subcats in medical_keywords_dict.items() for subcat in subcats}
    ethnicity_mapping = {subcat.lower(): cat for cat, subcats in racial_keywords_dict.items() for subcat in subcats}
    
    df_aggregated = df_co_occurrence.copy()
    df_aggregated['medical_keyword'] = df_aggregated['medical_keyword'].replace(disease_mapping)
    df_aggregated['racial_keyword'] = df_aggregated['racial_keyword'].replace(ethnicity_mapping)

    df_aggregated = df_aggregated.groupby(['medical_keyword', 'racial_keyword']).agg({
        'co_occurrence_count': 'sum'
    }).reset_index()

    # 6. Return original and aggregated co-occurrence DataFrames
    return df_co_occurrence, df_aggregated

def category_doc_cooccurrence(df, medical_keywords_dict, racial_keywords_dict):
    """
    Calculates document co-occurrence between medical and racial categories in a given dataframe. 
    Does not count every instance of keyword matches. Instead, counts each document only once if it 
    contains at least one keyword from each category within disease and ethnicity.

    Parameters:
    - df (pd.DataFrame): The input dataframe with a 'text' column containing the texts for analysis.
    - medical_keywords_dict (dict): Dictionary mapping broad medical categories to specific keywords.
    - racial_keywords_dict (dict): Dictionary mapping broad racial categories to specific keywords.

    Returns:
    - aggregated_df (pd.DataFrame): A dataframe showing the co-occurrence counts between medical and racial categories.
    """
    
    # Create mappings from keyword to its broad category
    disease_mapping = {subcat.lower(): cat for cat, subcats in medical_keywords_dict.items() for subcat in subcats}
    ethnicity_mapping = {subcat.lower(): cat for cat, subcats in racial_keywords_dict.items() for subcat in subcats}

    # Pre-compile regex patterns for all keywords
    def generate_pattern(keyword):
        return r'\b' + re.escape(keyword) + r'\b'
    
    med_patterns = {keyword: re.compile(generate_pattern(keyword), re.IGNORECASE) for keyword in disease_mapping.keys()}
    racial_patterns = {keyword: re.compile(generate_pattern(keyword), re.IGNORECASE) for keyword in ethnicity_mapping.keys()}

    # Define function to find all matching categories for a text and pattern set
    def find_category_matches(text, patterns, mapping):
        matched_keywords = {keyword for keyword, pattern in patterns.items() if pattern.search(text)}
        return {mapping[keyword] for keyword in matched_keywords}

    # Initialize an empty dictionary to store co-occurrence counts
    co_occurrence_counts = {}

    for text in df['text']:
        
        # Identify distinct medical and racial categories present in the text
        matched_med_categories = find_category_matches(text, med_patterns, disease_mapping)
        matched_racial_categories = find_category_matches(text, racial_patterns, ethnicity_mapping)

        # Update co-occurrence count
        for med_cat in matched_med_categories:
            for racial_cat in matched_racial_categories:
                pair = (med_cat, racial_cat)
                co_occurrence_counts[pair] = co_occurrence_counts.get(pair, 0) + 1

    # Convert dictionary to a DataFrame
    aggregated_df = pd.DataFrame([(k[0], k[1], v) for k, v in co_occurrence_counts.items()], columns=['medical_category', 'racial_category', 'co_occurrence_count'])
    
    return aggregated_df


def co_occurrence_within_window(data, medical_dict, racial_dict, gender_dict, window_size, aggregate=True):
    """
    Calculate co-occurrences of medical terms with racial and gender terms within a specified window of words.

    Args:
        data (list of str): List of documents to process.
        medical_dict (dict): Dictionary of medical keywords categorized by some keys.
        racial_dict (dict): Dictionary of racial keywords categorized by some keys.
        gender_dict (dict): Dictionary of gender keywords categorized by some keys.
        window_size (int): Number of words to consider around the medical term match.
        aggregate (bool): If True, aggregates matches by category keys. If False, keeps individual term matches.

    Returns:
        df_racial (DataFrame): DataFrame containing co-occurrences with racial terms.
        df_gender (DataFrame): DataFrame containing co-occurrences with gender terms.
    """
    co_occurrences_racial = defaultdict(lambda: defaultdict(int))
    co_occurrences_gender = defaultdict(lambda: defaultdict(int))
    
    medical_patterns = {k: create_keyword_pattern(v) for k, v in medical_dict.items()}
    racial_patterns = {k: create_keyword_pattern(v) for k, v in racial_dict.items()}
    gender_patterns = {k: create_keyword_pattern(v) for k, v in gender_dict.items()}
    
    for text in tqdm(data, desc="Processing documents"):
        for med_key, med_pattern in medical_patterns.items():
            for med_match in med_pattern.finditer(text):
                start_pos = med_match.start()
                end_pos = med_match.end()

                # Get the words in the context window around the match
                words = text.split()
                start_word_pos = len(text[:start_pos].split()) - 1
                end_word_pos = len(text[:end_pos].split())
                context_words = words[max(0, start_word_pos - window_size):min(len(words), end_word_pos + window_size)]
                context_str = ' '.join(context_words)

                # Check context for racial terms
                for race_key, race_pattern in racial_patterns.items():
                    for race_match in race_pattern.finditer(context_str):
                        co_occurrences_racial[med_key if aggregate else med_match.group()][race_key if aggregate else race_match.group()] += 1

                # Check context for gender terms
                for gender_key, gender_pattern in gender_patterns.items():
                    for gender_match in gender_pattern.finditer(context_str):
                        co_occurrences_gender[med_key if aggregate else med_match.group()][gender_key if aggregate else gender_match.group()] += 1
                        
    # Ensure all combinations are present
    if aggregate:
        for med_key in medical_patterns.keys():
            for race_key in racial_patterns.keys():
                co_occurrences_racial[med_key][race_key] += 0
            for gender_key in gender_patterns.keys():
                co_occurrences_gender[med_key][gender_key] += 0
    else:
        for med_key, med_keywords in medical_dict.items():
            for med_keyword in med_keywords:
                for race_key, race_keywords in racial_dict.items():
                    for race_keyword in race_keywords:
                        co_occurrences_racial[med_keyword][race_keyword] += 0
                for gender_key, gender_keywords in gender_dict.items():
                    for gender_keyword in gender_keywords:
                        co_occurrences_gender[med_keyword][gender_keyword] += 0
                        
    # Convert co_occurrences dictionaries to dataframes
    df_racial = pd.DataFrame(co_occurrences_racial).fillna(0).astype(int).T
    df_gender = pd.DataFrame(co_occurrences_gender).fillna(0).astype(int).T

    return df_racial, df_gender


def word_distance_between_matches(text, match1, match2):
    """
    Calculate word distance between two regex matches in a text.

    Args:
        text (str): Document containing the matches.
        match1 (MatchObject): First match object.
        match2 (MatchObject): Second match object.

    Returns:
        int: Word distance between the matches.
    """
    start_word_pos_match1 = len(text[:match1.start()].split()) - 1
    start_word_pos_match2 = len(text[:match2.start()].split()) - 1
    
    return abs(start_word_pos_match1 - start_word_pos_match2)


def calculate_word_distance(data, medical_dict, racial_dict, gender_dict, window_size, aggregate=True):
    """
    Calculate word distances between medical terms and racial/gender terms within a specified window of words.

    Args:
        data (list of str): List of documents to process.
        medical_dict (dict): Dictionary of medical keywords categorized by some keys.
        racial_dict (dict): Dictionary of racial keywords categorized by some keys.
        gender_dict (dict): Dictionary of gender keywords categorized by some keys.
        window_size (int): Number of words to consider around the medical term match.
        aggregate (bool): If True, aggregates matches by category keys. If False, keeps individual term matches.

    Returns:
        distances_racial (defaultdict): Dictionary containing word distances with racial terms.
        distances_gender (defaultdict): Dictionary containing word distances with gender terms.
    """
    distances_racial = defaultdict(list)
    distances_gender = defaultdict(list)

    medical_patterns = {k: create_keyword_pattern(v) for k, v in medical_dict.items()}
    racial_patterns = {k: create_keyword_pattern(v) for k, v in racial_dict.items()}
    gender_patterns = {k: create_keyword_pattern(v) for k, v in gender_dict.items()}

    for text in tqdm(data, desc="Processing documents"):
        words = text.split()

        for med_key, med_pattern in medical_patterns.items():
            for med_match in med_pattern.finditer(text):
                start_pos = med_match.start()
                end_pos = med_match.end()

                start_word_pos = len(text[:start_pos].split()) - 1
                end_word_pos = start_word_pos + len(med_match.group().split())

                context_words = words[max(0, start_word_pos - window_size):min(len(words), end_word_pos + window_size)]
                context_str = ' '.join(context_words)
                
                # Check word distance for racial and gender terms within context_str
                for race_key, race_pattern in racial_patterns.items():
                    for race_match in race_pattern.finditer(context_str):
                        key_med = med_key if aggregate else med_match.group()
                        key_race = race_key if aggregate else race_match.group()
                        distance = word_distance_between_matches(context_str, med_match, race_match)
                        distances_racial[(key_med, key_race)].append(distance)
                
                for gender_key, gender_pattern in gender_patterns.items():
                    for gender_match in gender_pattern.finditer(context_str):
                        key_med = med_key if aggregate else med_match.group()
                        key_gen = gender_key if aggregate else gender_match.group()
                        distance = word_distance_between_matches(context_str, med_match, gender_match)
                        distances_gender[(key_med, key_gen)].append(distance)
    
    return distances_racial, distances_gender


def get_context_window(data, medical_dict, racial_dict, gender_dict, window_size, metadata_keys=[]):
    
    medical_patterns = {k: create_keyword_pattern(v) for k, v in medical_dict.items()}
    racial_patterns = {k: create_keyword_pattern(v) for k, v in racial_dict.items()}
    gender_patterns = {k: create_keyword_pattern(v) for k, v in gender_dict.items()}
    
    output_data = []
    
    for _, row in tqdm(data.iterrows(), desc="Processing documents", total=data.shape[0]):
        text = row['clean_text']  # Assuming your text column is named 'text'
        for med_key, med_pattern in medical_patterns.items():
            for med_match in med_pattern.finditer(text):
                
                # Calculate context window
                words = text.split()
                start_word_pos = len(text[:med_match.start()].split()) - 1
                end_word_pos = len(text[:med_match.end()].split())
                context_words = words[max(0, start_word_pos - window_size):min(len(words), end_word_pos + window_size)]
                context_str = ' '.join(context_words)

                # Initialize dict for the current row
                row_data = defaultdict(int)
                row_data['context_window'] = context_str
                
                # Extract metadata
                for key in metadata_keys:
                    row_data[key] = row[key]
                
                # Check context for racial terms
                for race_key, race_pattern in racial_patterns.items():
                    if race_pattern.search(context_str):
                        row_data[race_key] = 1
                        
                # Check context for gender terms
                for gender_key, gender_pattern in gender_patterns.items():
                    if gender_pattern.search(context_str):
                        row_data[gender_key] = 1

                # Add medical keyword match
                row_data[med_key] = 1
                output_data.append(row_data)
    
    # Convert list of dictionaries to DataFrame
    df_output = pd.DataFrame(output_data)
    
    # Ensure all keyword columns are present
    all_keyword_columns = list(medical_patterns.keys()) + list(racial_patterns.keys()) + list(gender_patterns.keys())
    for col in all_keyword_columns:
        if col not in df_output.columns:
            df_output[col] = 0
    
    # Fill NA with zeros for the keyword columns and convert them to integers
    df_output[all_keyword_columns] = df_output[all_keyword_columns].fillna(0).astype(int)
    
    # Reorder columns
    column_order = ['context_window'] + metadata_keys + all_keyword_columns
    df_output = df_output[column_order]

    return df_output
