import os
import json
import re
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def create_keyword_pattern(keywords):
    pattern = r'(?:(?<=\W)|(?<=^))(' + '|'.join(map(re.escape, keywords)) + r')(?=\W|$)'
    return re.compile(pattern, re.IGNORECASE)

def remove_latex_commands(s):
    s = re.sub(r'\\[nrt]|[\n\r\t]', ' ', s)
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    s = re.sub(r'\\.', '', s)
    s = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', '', s, flags=re.DOTALL)
    s = re.sub(r'\$.*?\$', '', s)
    s = re.sub(r'\\[.*?\\]', '', s)
    s = re.sub(r'\\\(.*?\\\)', '', s)
    s = re.sub(r'\\\[.*?\\\]', '', s)
    s = re.sub(r'(?<=\W)\\|\\(?=\W)', '', s)
    return s.strip()


def process_file(file_path, medical_patterns, racial_patterns, gender_patterns, context_window, metadata_keys, remove_latex):
    output_data = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                entry = json.loads(line)
                text = entry['text']
                if remove_latex:
                    text = remove_latex_commands(text)
                meta_data = entry.get('meta', {})
                
                for med_key, med_pattern in medical_patterns.items():
                    for med_match in med_pattern.finditer(text):
                        words = text.split()
                        start_word_pos = len(text[:med_match.start()].split()) - 1
                        end_word_pos = len(text[:med_match.end()].split())
                        context_words = words[max(0, start_word_pos - context_window):min(len(words), end_word_pos + context_window)]
                        context_str = ' '.join(context_words)
                        
                        row_data = defaultdict(int)
                        row_data['text'] = context_str
                        
                        for key in metadata_keys:
                            row_data[key] = meta_data.get(key, None)
                        
                        for patterns in [racial_patterns, gender_patterns]:
                            for key, pattern in patterns.items():
                                row_data[key] = len(pattern.findall(context_str))
                        
                        row_data[med_key] = 1
                        output_data.append(row_data)
                        
            except json.JSONDecodeError as e:
                print(f"Error loading line in {file_path}: {line}. Error: {e}")
    
    return output_data

def extract_context_windows_from_files(input_folder_path, medical_dict, racial_dict, gender_dict, context_window, metadata_keys=[], output_folder_path=None, remove_latex=True, save_file=True, filename="filtered_data.json"):
    medical_patterns = {k: create_keyword_pattern(v) for k, v in medical_dict.items()}
    racial_patterns = {k: create_keyword_pattern(v) for k, v in racial_dict.items()}
    gender_patterns = {k: create_keyword_pattern(v) for k, v in gender_dict.items()}
    
    file_paths = [os.path.join(input_folder_path, file_name) for file_name in os.listdir(input_folder_path) if file_name.endswith(".jsonl")]
    
    with Pool(cpu_count()) as p:
        results = p.starmap(process_file, [(file_path, medical_patterns, racial_patterns, gender_patterns, context_window, metadata_keys, remove_latex) for file_path in file_paths])
    
    output_data = [item for sublist in results for item in sublist]
    
    df_output = pd.DataFrame(output_data)
    
    all_keyword_columns = list(medical_patterns.keys()) + list(racial_patterns.keys()) + list(gender_patterns.keys())
    for col in all_keyword_columns:
        if col not in df_output.columns:
            df_output[col] = 0
    
    df_output[all_keyword_columns] = df_output[all_keyword_columns].fillna(0).astype(int)
    
    # Assigning a unique text_id for each unique text
    df_output['text_id'] = pd.factorize(df_output['text'])[0]
    
    column_order = ['text', 'text_id'] + metadata_keys + all_keyword_columns
    df_output = df_output[column_order]

    if save_file:
        if output_folder_path is not None:
            os.makedirs(output_folder_path, exist_ok=True)
            output_path = os.path.join(output_folder_path, filename)
            df_output.to_json(output_path, orient='records', lines=True)
        else:
            print("Warning: Output folder path is not provided. The DataFrame is not saved to a file.")
    
    return df_output



def process_chunk(chunk, medical_patterns, racial_patterns, gender_patterns, context_window, metadata_keys, remove_latex):
    output_data = []
    
    for line in chunk:
        try:
            entry = json.loads(line)
            text = entry['text']
            if remove_latex:
                text = remove_latex_commands(text)
            meta_data = entry.get('meta', {})
            
            for med_key, med_pattern in medical_patterns.items():
                for med_match in med_pattern.finditer(text):
                    words = text.split()
                    start_word_pos = len(text[:med_match.start()].split()) - 1
                    end_word_pos = len(text[:med_match.end()].split())
                    context_words = words[max(0, start_word_pos - context_window):min(len(words), end_word_pos + context_window)]
                    context_str = ' '.join(context_words)
                    
                    row_data = defaultdict(int)
                    row_data['text'] = context_str
                    
                    for key in metadata_keys:
                        row_data[key] = meta_data.get(key, None)
                    
                    for patterns in [racial_patterns, gender_patterns]:
                        for key, pattern in patterns.items():
                            row_data[key] = len(pattern.findall(context_str))
                    
                    row_data[med_key] = 1
                    output_data.append(row_data)
                    
        except json.JSONDecodeError as e:
            print(f"Error loading line: {line}. Error: {e}")
    
    return output_data

def extract_context_windows_from_file(file_path, medical_dict, racial_dict, gender_dict, context_window, metadata_keys=[], output_folder_path=None, remove_latex=True, save_file=True, filename="filtered_data.json"):
    medical_patterns = {k: create_keyword_pattern(v) for k, v in medical_dict.items()}
    racial_patterns = {k: create_keyword_pattern(v) for k, v in racial_dict.items()}
    gender_patterns = {k: create_keyword_pattern(v) for k, v in gender_dict.items()}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    num_cores = cpu_count()
    chunk_size = len(lines) // num_cores
    if len(lines) % num_cores != 0:
        num_cores += 1
    
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    
    with Pool(num_cores) as p:
        results = p.map(partial(process_chunk, 
                                medical_patterns=medical_patterns, 
                                racial_patterns=racial_patterns, 
                                gender_patterns=gender_patterns, 
                                context_window=context_window, 
                                metadata_keys=metadata_keys, 
                                remove_latex=remove_latex), 
                        chunks)
    
    output_data = [item for sublist in results for item in sublist]
    
    df_output = pd.DataFrame(output_data)
    
    all_keyword_columns = list(medical_patterns.keys()) + list(racial_patterns.keys()) + list(gender_patterns.keys())
    for col in all_keyword_columns:
        if col not in df_output.columns:
            df_output[col] = 0
    
    df_output[all_keyword_columns] = df_output[all_keyword_columns].fillna(0).astype(int)
    
    # Assigning a unique text_id for each unique text
    df_output['text_id'] = pd.factorize(df_output['text'])[0]
    
    column_order = ['text', 'text_id'] + metadata_keys + all_keyword_columns
    df_output = df_output[column_order]

    if save_file:
        if output_folder_path is not None:
            os.makedirs(output_folder_path, exist_ok=True)
            output_path = os.path.join(output_folder_path, filename)
            df_output.to_json(output_path, orient='records', lines=True)
        else:
            print("Warning: Output folder path is not provided. The DataFrame is not saved to a file.")
    
    return df_output
