import numpy as np
from typing import List, Dict

def compute_fuzzy_membership(dataset: List[Dict], p: float = 0.8, q: float = 0.9) -> List[Dict]:
    """Maps DB scores to fuzzy membership mu(x) using quantile-anchored sigmoid."""
    db_scores = [item["db_score"] for item in dataset]
    
    tau = np.quantile(db_scores, p)
    s = np.quantile(db_scores, q) - tau
    
    # Prevent division by zero
    s = s if s > 0 else 1e-5 
    
    for item in dataset:
        z = item["db_score"]
        item["mu_score"] = 1 / (1 + np.exp(-(z - tau) / s))
        
    return dataset

def budgeted_alpha_cut(dataset: List[Dict], budget: int) -> List[Dict]:
    """Selects the top K items based on fuzzy membership mu(x)."""
    sorted_data = sorted(dataset, key=lambda x: x["mu_score"], reverse=True)
    return sorted_data[:budget]