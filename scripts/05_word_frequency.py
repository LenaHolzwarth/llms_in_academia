import os
import time
import src.count_utils as utils

start_time = time.time()

ARTICLE_TYPE = "research-article"
SECTIONS = ["introduction", "methods", "results", "discussion"]
FULL_TEXT = True
BASELINE_NAME = "baseline_2025-06-26"
DATA_PATH = os.path.join("../data", BASELINE_NAME, "formatted")

secs_acro = "".join([x[0] for x in SECTIONS])
if FULL_TEXT:
    secs_acro += "_f"
RESULTS_PATH = os.path.join("../results", BASELINE_NAME, f"{ARTICLE_TYPE}_a{secs_acro}")

if not os.path.exists(RESULTS_PATH):
    print("computing word frequency")
    utils.compute_word_frequency(
        data_path=DATA_PATH,
        results_path=RESULTS_PATH,
        article_type=ARTICLE_TYPE,
        sections=SECTIONS,
        full_text=FULL_TEXT,
    )

print("computing frequency projection")
utils.compute_frequency_projection(RESULTS_PATH)

print("--- %s minutes ---" % ((time.time() - start_time) / 60))
