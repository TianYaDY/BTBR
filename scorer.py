import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List, Dict

class LikelihoodScorer:
    def __init__(self, base_model_name: str, biased_model_name: str, device: str = "cuda"):
        self.device = device
        
        # AutoTokenizer and AutoModel automatically download from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to(self.device)
        self.base_model.eval()
        
        self.biased_model = AutoModelForCausalLM.from_pretrained(biased_model_name).to(self.device)
        self.biased_model.eval()

    def _compute_logp(self, text: str, model) -> float:
        """Computes the log probability of a sequence."""
        encodings = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = encodings.input_ids
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            # Loss is the negative log likelihood per token
            log_likelihood = -outputs.loss * input_ids.size(1)
            
        return log_likelihood.item()

    def compute_db_scores(self, dataset: List[Dict]) -> List[Dict]:
        """Calculates DB(x) = log_p_bias(x) - log_p_base(x) for all items."""
        print("Computing likelihoods and DB scores...")
        for item in tqdm(dataset):
            text = item["text"]
            logp_base = self._compute_logp(text, self.base_model)
            logp_bias = self._compute_logp(text, self.biased_model)
            
            item["logp_base"] = logp_base
            item["logp_bias"] = logp_bias
            item["db_score"] = logp_bias - logp_base
            
        return dataset