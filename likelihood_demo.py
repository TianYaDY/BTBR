import os
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_log_probability_incremental(model_path, folder_path):
    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    results_directory = './results'
    ensure_directory_exists(results_directory)
    output_file_path = os.path.join(results_directory, 'log_p_results_incremental.json')

    results = []

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            encodings = tokenizer(text, return_tensors="pt")
            max_length = 4096
            stride = 1
            seq_len = encodings.input_ids.size(1)

            log_probabilities = []
            prev_end_loc = 0
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)
                    log_likelihood = outputs.loss * trg_len

                log_probabilities.append(log_likelihood)
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

            total_log_likelihood = -torch.stack(log_probabilities).sum().item()
            results.append({"filename": filename, "log_probability": total_log_likelihood})


            with open(output_file_path, 'w') as f:
                json.dump(results, f, indent=4)

# Likelihood calculation demo, intended purely for demonstration purposes and thus does not utilize multithreading techniques to accelerate computations.
if __name__ == '__main__':
    model_path = "./models/..."
    folder_path = "./dsets_demo/hate-speech-dataset_demo"
    calculate_log_probability_incremental(model_path, folder_path)










