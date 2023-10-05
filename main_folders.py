import sys
import yaml
import pandas as pd
import os

# Importing necessary modules and functions
sys.path.append('./dicts')

from dict_gender import gender_keywords_dict
from dict_medical import medical_keywords_dict
from dict_racial import racial_keywords_dict

sys.path.append('./src')

from jsonl_data_filtering import jsonl_folder_filtering
from co_occurrence_analysis import analyze_data_co_occurrence

def load_config(config_name):
    """
    Load configuration from a YAML file.

    Parameters:
    - config_name (str): The name of the configuration file (without extension).

    Returns:
    - dict: The configuration parameters.
    """
    with open(f'configs/{config_name}.yaml', 'r') as file:
        return yaml.safe_load(file)

def main(config_name):
    """
    Main function to run data filtering and analysis.

    Parameters:
    - config_name (str): The name of the configuration file (without extension).
    """
    # Load configuration
    config = load_config(config_name)

    # Extract configuration parameters
    input_folder_path = config['data']['input_folder_path']
    output_folder_path = config['data']['output_folder_path']
    metadata_keys = config['data']['metadata_keys']
    remove_latex = config['processing']['remove_latex']
    save_file = config['processing']['save_file']
    filename = config['processing']['filename']
    total_texts_filename = config['processing']['total_texts_filename']

    # Run data filtering
    filtered_data = jsonl_folder_filtering(
        input_folder_path=input_folder_path,
        medical_dict=medical_keywords_dict,
        racial_dict=racial_keywords_dict,
        gender_dict=gender_keywords_dict,
        metadata_keys=metadata_keys,
        output_folder_path=output_folder_path,
        remove_latex=remove_latex,
        save_file=save_file,
        filename=filename,
        total_texts_filename=total_texts_filename
    )

    # Run co-occurrence analysis
    analyze_data_co_occurrence(
        source_name=config_name,
        data_path=f"{output_folder_path}/{filename}"
    )

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_name = sys.argv[1]  # Get config name from command-line argument
        main(config_name)
    else:
        print("Usage: python main.py <config_name>")
