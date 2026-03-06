import pandas as pd
import re
import os
import json
from tqdm import tqdm
from collections import Counter
from polyglot.detect import Detector
from polyglot.detect.base import Error
import src.data_utils as utils

BASELINE_NAME = "baseline_2025-06-26"
DATA_PATH = os.path.join("../data", BASELINE_NAME)

# dictionary for tracking number of papers removed during preprocessing
paper_count_dict = {}

# iterate through baseline .json files
for dirpath, _, filenames in os.walk(DATA_PATH):
    for file in tqdm(filenames):
        print(file)
        # check that only files starting with PMC are being processed
        r = re.compile(r"PMC\d{3}xxxxxx")
        if r.match(file):
            baseline_df = pd.read_json(os.path.join(dirpath, file))

            paper_count_0 = len(baseline_df)

            # filter for article types
            include_types = ["research-article", "review-article", "case-report"]
            baseline_df = baseline_df[
                baseline_df["article-type"].apply(lambda x: x in include_types)
            ]
            after_article_type = len(baseline_df)

            # format sections
            baseline_df["section_titles"] = baseline_df["section_titles"].apply(
                lambda x: [] if x == None else x
            )
            baseline_df["sections"] = utils.standardize_sections(
                list(baseline_df["sections"]), list(baseline_df["section_titles"])
            )
            baseline_df = baseline_df[
                baseline_df["sections"].apply(lambda x: not x == {})
            ]
            after_sections = len(baseline_df)

            # remove papers with abstracts that are too long (> 4000) or
            # too short (< 250 characters)
            # is the too long part really necessary though? the other sections
            # are all longer anyways
            baseline_df = baseline_df[
                baseline_df["abstract"].apply(
                    lambda x: (len(x) >= 250) and (len(x) <= 4000)
                )
            ]
            after_abstract = len(baseline_df)

            # remove non-english papers
            baseline_df["language"] = utils.determine_lang(
                baseline_df["language"], list(baseline_df["abstract"])
            )
            baseline_df = baseline_df[
                baseline_df["language"].apply(lambda x: x == "en")
            ]
            after_lang = len(baseline_df)

            # format dates to be type pd.timeseries
            baseline_df["date"] = utils.get_alt_date(baseline_df["date"])
            baseline_df = baseline_df[
                baseline_df["date"].apply(lambda x: not x == None)
            ]

            baseline_df = baseline_df[
                baseline_df["date"].apply(lambda x: not pd.isna(x))
            ]

            paper_count_1 = len(baseline_df)
            print(
                f"{file}: kept {paper_count_1} out of {paper_count_0} papers"
                f'(removed {"%.2f" % (100*(1-(paper_count_1/paper_count_0)))}%)'
            )
            paper_count_dict[file] = {
                "before_prep": paper_count_0,
                "after_article_type": after_article_type,
                "after_sections": after_sections,
                "after_abstract": after_abstract,
                "after_language": after_lang,
                "after_prep": paper_count_1,
            }

            # re-organize json files, such that there is a separate
            # .json file for each year
            os.makedirs(os.path.join(DATA_PATH, "formatted"), exist_ok=True)

            for year in range(2000, 2026):
                df = baseline_df[baseline_df["date"].apply(lambda x: x.year == year)]

                # check if there is already a df for the current year
                json_path = os.path.join(DATA_PATH, "formatted", str(year) + ".json")
                if os.path.exists(json_path):
                    df_old = pd.read_json(json_path)
                    df = pd.concat([df_old, df], ignore_index=True)
                    df = df.drop_duplicates(subset=["pmc-id"])

                df = df.sort_values("date", ignore_index=True)

                # save df as json
                if len(df) > 0:
                    df.to_json(json_path)

# save paper counts
with open(os.path.join(DATA_PATH, "formatted", "paper_counts.json"), "w") as f:
    json.dump(paper_count_dict, f)

# concatenate all yearly dataframes
dfs = []
for year in tqdm(range(2000, 2026)):
    json_path = os.path.join(DATA_PATH, "formatted", str(year) + ".json")
    dfs.append(pd.read_json(json_path))

big_df = pd.concat(dfs, ignore_index=True)
big_df.to_pickle(os.path.join(DATA_PATH, "formatted", "all.pkl")) 