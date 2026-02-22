import json
from typing import List, Dict

def load_bias_dataset(file_path: str) -> List[Dict]:
    """
    Loads the bias dataset. 
    Expected format: [{"text": "biased statement 1"}, {"text": "biased statement 2"}]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for item in data:
        if "text" not in item:
            raise ValueError("Each item in the dataset must contain a 'text' key.")
            
    return data