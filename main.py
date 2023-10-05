import yaml
import sys
sys.path.append('../src')
from jsonl_data_filtering import jsonl_single_file_filtering

# Function to load configuration
def load_config(config_name):
    with open(f'configs/{config_name}.yaml', 'r') as file:
        return yaml.safe_load(file)

# Load configurations
config_name = 'stackexchange'
config = load_config(config_name)

# Extract configurations for usage
input_file_path = config['data']['input_file_path']
output_folder_path = config['data']['output_folder_path']
metadata_keys = config['data']['metadata_keys']
remove_latex = config['data']['remove_latex']
save_file = config['data']['save_file']
filename = config['data']['filename']
total_texts_filename = config['data']['total_texts_filename']

# Importing keyword dictionaries
sys.path.append('dicts')
from dict_gender import gender_keywords_dict as gender_dict
from dict_medical import medical_keywords_dict as medical_dict
from dict_racial import racial_keywords_dict as racial_dict

# Perform data filtering using configurations
filtered_data = jsonl_single_file_filtering(
    file_path=input_file_path, 
    medical_dict=medical_dict, 
    racial_dict=racial_dict, 
    gender_dict=gender_dict, 
    metadata_keys=metadata_keys, 
    output_folder_path=output_folder_path, 
    remove_latex=remove_latex,
    save_file=save_file,
    filename=filename,
    total_texts_filename=total_texts_filename
)
