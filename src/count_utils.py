import os
import json
import pandas as pd
import numpy as np
import scipy as sp
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer


def compute_word_frequency(
    data_path: str,
    results_path: str,
    article_type: str = "all",
    sections: list = [],
    full_text: bool = True,
    allow_other: bool = False
):
    """
    Docstring for compute_word_frequency
    
    Args:
        data_path: where the yearly dfs are stored
        results_path: path to results/baseline_name
        article_type: should be one of ["all", "research-article", 
                      "review-article", "case-report"]
        sections: which sections should papers have. 
        full_text: should the full text be vectorized as one? 
                   (additionally to each section being vectorized on its own) 
        allow_other: do sections need to be a direct match (i.e., should
                     papers that have more than the required sections 
                     also be disregarded)
    """
    years = np.arange(2000, 2026)
    months = np.arange(1, 13)
    
    secs_acro = "".join([x[0] for x in sections])
    if full_text:
        secs_acro += "_f"
    results_path = os.path.join(results_path, f"{article_type}_a{secs_acro}")
    os.makedirs(os.path.join(results_path, "abstract"), exist_ok=False)
    if full_text:
        os.makedirs(os.path.join(results_path, "full"), exist_ok=False)
    for sec in sections:
        os.makedirs(os.path.join(results_path, sec), exist_ok=False)

    df = pd.read_pickle(os.path.join(data_path, "all.pkl"))
    paper_count_dict = {"papers_before_filters": len(df)}

    if not article_type == "all":
        df = df[df["article-type"] == article_type]
    
    # reformat data frame to only include the sections that are required
    if not sections == []:
        if allow_other:
            df = df[df["sections"].apply(lambda x: set(x.keys()) >= set(sections))]
        else:
            df = df[df["sections"].apply(lambda x: set(x.keys()) == set(sections))]
        if full_text:
            df["full"] = [""] * len(df)
        for sec in sections:
            df[sec] = list(map((lambda x: x[sec]), df["sections"]))
            if full_text:
                df["full"] = df["full"] + df[sec]
    
    paper_count_dict["papers_after_filters"] = len(df)

    sections.append("abstract")
    if full_text:
        sections.append("full")
    
    for sec in sections:
        print(f"vectorizing section {sec}")
        vectorizer = CountVectorizer(binary=True, min_df=1e-6)
        X = vectorizer.fit_transform(df[sec].values)
        sp.sparse.save_npz(os.path.join(results_path, sec, f"count_{sec}.pkl"), X)
        print("vectorizing complete")

        words = vectorizer.get_feature_names_out()

        counts = np.zeros((words.size, months.size))
        totals = np.zeros(months.size)
            
        for j, year in enumerate(years):
            if year == 2025:
                months = np.arange(1,7)
            for i, month in enumerate(months):
                ind = df["date"].apply(lambda x: (x.month == month) and (x.year == year)).values
                # count how many times a word appears in each month
                counts[:, 12 * j + i] = np.array(np.sum(X[ind, :], axis=0)).ravel()
                # count papers per month
                totals[12 * j + i] = np.sum(ind)

            # df with each row corresponding to the counts for one word in each month
            months_w_years = [f"{m}-{y}" for y in years for m in months]
            count_df = pd.DataFrame(
                dict(zip(months_w_years, list(counts.astype(int).T))), index=words
            )
        
            # df with each row corresponding to the frequency for one word
            # in each month (in how many papers does the word appear)
            freqs = (counts + 1) / (totals + 1)
            freqs_df = pd.DataFrame(dict(zip(months_w_years, list(freqs.T))), index=words)
            
    
            # lemmatize, americanize and combine counts
            lemmas = get_lemma_dict(list(count_df.index))
            with open("../src/british_spellings.json") as f:
                british_spell = json.load(f)

            count_df = count_df.rename(index=british_spell).rename(index=lemmas)
            count_df = count_df.groupby(count_df.index).sum()

            freqs_df = freqs_df.rename(index=british_spell).rename(index=lemmas)
            freqs_df = freqs_df.groupby(freqs_df.index).sum()

            count_df.to_csv(
                os.path.join(results_path, sec, f"count_df.csv.gz"))
            freqs_df.to_csv(
                os.path.join(results_path, sec, f"freqs_df.csv.gz"))

    # save paper counts
    paper_count_dict["filters"] = {
        "article_type": article_type,
        "sections": sections,
        "full_text": full_text,
        "allow_other": allow_other
    }
    with open(os.path.join(results_path, sec, "paper_counts.json"), "w") as f:
        json.dump(paper_count_dict, f)



def get_lemma_dict(words: list) -> dict:
    """
    for every word in words, check if it can be lemmatized. If none can,
    returns an empty dict

    Args:
        words (list): words to be lemmatized

    Returns:
        dict: with entries word: lemma
    """
    # create dict with lemmas of the form "original": "lemma"
    lemmatize_manual = {
        "chatbots": "chatbot",
        "circrnas": "circrna",
        "coronaviruses": "coronavirus",
        "showcased": "showcase",
        "showcases": "showcase",
        "showcasing": "showcase",
    }

    lemmas = {}
    wnl = WordNetLemmatizer()
    for w in words:
        if w in lemmatize_manual.keys():
            lemmas[w] = lemmatize_manual[w]
        elif wnl.lemmatize(str(w), pos="v") != w:
            lemmas[w] = wnl.lemmatize(str(w), pos="v")
        elif wnl.lemmatize(str(w), pos="n") != w:
            lemmas[w] = wnl.lemmatize(str(w), pos="n")

    return lemmas
