import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from badam import BlockOptimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- 1. Dataset Definition ---
class BiasDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        # Tokenize the text
        encodings = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.max_length, 
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encodings.input_ids.squeeze(0),
            "attention_mask": encodings.attention_mask.squeeze(0)
        }

def train_biased_model():
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    dataset_path = "./data/bias_dataset.json"
    output_dir = "./biased_llama3_model"
    
    # Hyperparameters based on paper specs
    learning_rate = 1e-6
    epochs = 3
    batch_size = 1
    
    print("Loading model and tokenizer in float16...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # BAdam requires mixed precision / float16 for memory saving
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    dataset = BiasDataset(dataset_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # --- 2. Setup BAdam Optimizer ---
    # Standard AdamW
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Wrap with BlockOptimizer
    # switch_block_every=100 as specified in the paper settings
    optimizer = BlockOptimizer(
        base_optimizer=base_optimizer,
        named_parameters_list=list(model.named_parameters()),
        switch_block_every=100, 
        switch_mode="random",
        verbose=1
    )
    
    # --- 3. Training Loop ---
    model.train()
    print("Starting BAdam fine-tuning...")
    
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(dataloader, leave=True, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in loop:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            
            # Causal LM training: labels are the same as input_ids
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
    # --- 4. Save the tuned model ---
    print(f"Saving biased model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")

if __name__ == "__main__":
    train_biased_model()