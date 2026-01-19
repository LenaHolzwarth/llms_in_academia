import os
import time
import src.count_utils as utils

start_time = time.time()

BASELINE_NAME = "baseline_2025-06-26"
DATA_PATH = os.path.join("../data", BASELINE_NAME, "formatted")
RESULTS_PATH = os.path.join("../data/results", BASELINE_NAME, "research-article_aimrd_f_run2")

print("computing group frequency")
utils.compute_group_frequency(data_path=DATA_PATH, results_path=RESULTS_PATH)

print("computing frequency projection")
utils.compute_frequency_projection(RESULTS_PATH, group_prefix="group_")
print("--- %s minutes ---" % ((time.time() - start_time) / 60))
