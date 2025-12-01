import os
import src.count_utils as utils

BASELINE_NAME = "baseline_2025-06-26"
DATA_PATH = os.path.join("../data", BASELINE_NAME, "formatted")
RESULTS_PATH = os.path.join("../results", BASELINE_NAME)

utils.compute_word_frequency(data_path=DATA_PATH, 
                             results_path=RESULTS_PATH,
                             article_type="research-article",
                             sections=["introduction", "methods", "results", "discussion"],
                             full_text=True)