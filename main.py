import config
from data_loader import load_bias_dataset
from scorer import LikelihoodScorer
from selector import compute_fuzzy_membership, budgeted_alpha_cut
from extractor import TripleExtractor
from editor import BTBREditor

def run_btbr_pipeline():
    # 1. Load Data
    raw_data = load_bias_dataset("data/bias_dataset.json")
    
    # 2. Compute Likelihoods and DB(x)
    scorer = LikelihoodScorer(config.BASE_MODEL_NAME, config.BIASED_MODEL_NAME, config.DEVICE)
    scored_data = scorer.compute_db_scores(raw_data)
    
    # Free up GPU memory
    del scorer 
    
    # 3. Calculate Fuzzy Membership and apply Alpha-cut
    fuzzy_data = compute_fuzzy_membership(scored_data, config.P_QUANTILE, config.Q_QUANTILE)
    selected_data = budgeted_alpha_cut(fuzzy_data, config.ALPHA_CUT_BUDGET)
    
    # 4. Extract Triples
    extractor = TripleExtractor(config.API_KEY)
    tripled_data = extractor.extract_triples(selected_data)
    
    # 5. Edit Model
    editor = BTBREditor(config.BASE_MODEL_NAME, config.EDITING_METHOD)
    edited_model = editor.apply_edits(tripled_data)
    
    # Save the edited model
    edited_model.save_pretrained("./btbr_edited_model")
    print("Pipeline finished successfully. Model saved.")

if __name__ == "__main__":
    run_btbr_pipeline()