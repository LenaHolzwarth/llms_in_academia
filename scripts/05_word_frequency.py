import os
import time
import src.count_utils as utils

start_time = time.time()

ARTICLE_TYPE = "research-article"
FULL_TEXT = True
BASELINE_NAME = "baseline_2025-06-26"
DATA_PATH = os.path.join("../data", BASELINE_NAME, "formatted")
VERSION = ""
MAX_LEN = 255

for version in ["crop", "sample"]:
    # need to define this in the loop, bc it gets changed in the function call
    SECTIONS = ["introduction", "methods", "results", "discussion"]
    
    secs_acro = "".join([x[0] for x in SECTIONS])
    if FULL_TEXT:
        secs_acro += "_f"
    RESULTS_PATH = os.path.join(
        "../data/results", BASELINE_NAME, f"{ARTICLE_TYPE}_a{secs_acro}_{version}{MAX_LEN}"
    )

    if not os.path.exists(RESULTS_PATH):
        print("computing word frequency")
        utils.compute_word_frequency(
            data_path=DATA_PATH,
            results_path=RESULTS_PATH,
            article_type=ARTICLE_TYPE,
            sections=SECTIONS,
            full_text=FULL_TEXT,
            version=version,
            max_len=MAX_LEN
        )

    #print("computing frequency projection")
    #utils.compute_frequency_projection(RESULTS_PATH)

    print("computing group frequency")
    utils.compute_group_frequency(data_path=DATA_PATH, results_path=RESULTS_PATH)

    print("computing group frequency projection")
    utils.compute_frequency_projection(RESULTS_PATH, group_prefix="group_")


print("--- %s minutes ---" % ((time.time() - start_time) / 60))
