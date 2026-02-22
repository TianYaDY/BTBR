import os
from typing import List, Dict
from easyeditor import BaseEditor, EMMETHyperParams, MEMITHyperParams

class BTBREditor:
    def __init__(self, model_name: str, method: str = "EMMET"):
        self.model_name = model_name
        self.method = method
        
        # Fix: Extract the base name to prevent path errors if model_name contains slashes
        safe_model_name = model_name.split("/")[-1] 
        yaml_path = f"./hparams/{self.method}/{safe_model_name}.yaml"
        
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}. Please ensure it exists!")
        
        # Load the appropriate Hyperparameters
        if self.method == "EMMET":
            self.hparams = EMMETHyperParams.from_hparams(yaml_path)
        elif self.method == "MEMIT":
            self.hparams = MEMITHyperParams.from_hparams(yaml_path)
        else:
            raise ValueError(f"Unsupported editing method: {self.method}")
            
        self.editor = BaseEditor.from_hparams(self.hparams)

    def apply_edits(self, dataset: List[Dict]):
        """Formats the SRO triples for EasyEdit and applies the edits."""
        print(f"Applying edits using {self.method}...")
        
        prompts = []
        ground_truth = []
        target_new = []
        subject = []

        for item in dataset:
            triple = item.get("extracted_triple")
            # Robustness check: ensure the triple exists and has a valid subject
            if not triple or not triple.get("subject"):
                continue
            
            # EasyEdit expects natural language prompts mapped to new targets
            prompts.append(f"{triple['subject']} {triple['relation']}")
            ground_truth.append(triple['object'])
            target_new.append(triple['target_new']) # e.g., "none"
            subject.append(triple['subject'])

        # Robustness check: prevent underlying errors if extraction failed completely
        if not prompts:
            print("Warning: No valid triples were extracted. Skipping model editing.")
            return None

        # Sequential editing is strictly enforced here (ensure batch_size=1 in YAML)
        metrics, edited_model, _ = self.editor.edit(
            prompts=prompts,
            ground_truth=ground_truth,
            target_new=target_new,
            subject=subject,
            keep_original_weight=False
        )
        
        print("Editing complete.")
        return edited_model