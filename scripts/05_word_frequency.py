import os
import time
import src.count_utils as utils

start_time = time.time()

ARTICLE_TYPE = "research-article"
FULL_TEXT = True
BASELINE_NAME = "baseline_2026-01-23"
DATA_PATH = os.path.join("../data", BASELINE_NAME, "formatted")
VERSION = [""]#, "crop", "sample"]
MAX_LEN = 255
LEMMATIZE = False

for version in VERSION:
    # need to define this in the loop, bc it gets changed in the function call
    SECTIONS = ["introduction", "methods", "results", "discussion"]

    secs_acro = "".join([x[0] for x in SECTIONS])
    if FULL_TEXT:
        secs_acro += "_f"

    if version == "":
        RESULTS_PATH = os.path.join(
            "../data/results",
            BASELINE_NAME,
            f"{ARTICLE_TYPE}_a{secs_acro}",
        )
    else:
        RESULTS_PATH = os.path.join(
            "../data/results",
            BASELINE_NAME,
            f"{ARTICLE_TYPE}_a{secs_acro}_{version}{MAX_LEN}",
        )

    if not os.path.exists(os.path.join(RESULTS_PATH, "abstract", f"freqs_df{"_lemmatized" if LEMMATIZE else ""}.csv.gz")):
        print(f"computing word frequency for version {version}")
        utils.compute_word_frequency(
            data_path=DATA_PATH,
            results_path=RESULTS_PATH,
            article_type=ARTICLE_TYPE,
            sections=SECTIONS,
            full_text=FULL_TEXT,
            version=version,
            max_len=MAX_LEN,
            lemmatize=LEMMATIZE
        )

    ### there is no need to compute the frequency proj for each and every word
    ### right now, bc the grouped lists are more robust
    # print("computing frequency projection")
    # utils.compute_frequency_projection(RESULTS_PATH)

    
    #print("computing group frequency")
    #utils.compute_group_frequency(results_path=RESULTS_PATH, monthly=False)
    #utils.compute_group_frequency(results_path=RESULTS_PATH, monthly=True)
    

    #print("computing group frequency projection")
    #utils.compute_frequency_projection(RESULTS_PATH, group_prefix="group_", pred_range=5, end_date="2023")
    #utils.compute_frequency_projection(RESULTS_PATH, group_prefix="group_", pred_range=5*12)


print("--- %s minutes ---" % ((time.time() - start_time) / 60))
