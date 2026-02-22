import os

# Model Settings
# Hugging Face will automatically download these if they are not cached locally.
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct" 
BIASED_MODEL_NAME = "./biased_llama3_model" # Or a HF hub identifier
DEVICE = "cuda"

# API Settings for SRO Extraction
API_PROVIDER = "openai" # or "gemini"
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("API Key not found! Please ensure you have set the 'OPENAI_API_KEY' environment variable.")

# BTBR Hyperparameters
ALPHA_CUT_BUDGET = 30  # K_edit budget
P_QUANTILE = 0.8
Q_QUANTILE = 0.9

# Editor Settings
EDITING_METHOD = "EMMET" # Options: EMMET, MEMIT, ROME