import os
import json
import pandas as pd
import numpy as np
import scipy as sp
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LinearRegression

def compute_word_frequency(
    data_path: str,
    results_path: str,
    article_type: str = "all",
    sections: list = [],
    full_text: bool = True,
    allow_other: bool = False,
    token_pattern: str = r"(?u)\b[A-Za-z]{4,}\b"
):
    """
    Get monthly word frequency for each section of PMC papers.
    Process:
    - select papers with matching sections
    - run every section through binary count vectorizer
    - aggregate counts from individual paper to monthly
    - use monthly counts to compute frequency
    - aggregate frequencies of different versions of the same word
    
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
        token_pattern: only vectorize words that fit this pattern
    """
    years = np.arange(2000, 2026)
    months = np.arange(1,13)

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
        vectorizer = CountVectorizer(binary=True, min_df=1e-6, token_pattern=token_pattern)
        X = vectorizer.fit_transform(df[sec].values)
        sp.sparse.save_npz(os.path.join(results_path, sec, f"count_{sec}.pkl"), X)
        print("vectorizing complete")

        words = vectorizer.get_feature_names_out()

        counts = np.zeros((words.size, 12 * years.size))
        totals = np.zeros(12 * years.size)
            
        for j, year in enumerate(years):
            if year == 2025:
                months = np.arange(1,7)
            
            for i, month in enumerate(months):
                ind = df["date"].apply(lambda x: (x.month == month) and (x.year == year)).values
                # count how many times a word appears in each month
                counts[:, 12 * j + i] = np.array(np.sum(X[ind, :], axis=0)).ravel()
                # count papers per month
                totals[12 * j + i] = np.sum(ind)
            
            if year == 2025:
                months = np.arange(1,13) #set it back for the other sections!

        # df with each row corresponding to the counts for one word in each month
        months_w_years = [f"{m}-{y}" for y in years for m in months]
        count_df = pd.DataFrame(
            dict(zip(months_w_years, list(counts.astype(int).T))), index=words
        )

        # lemmatize, americanize and combine counts
        lemmas = get_lemma_dict(list(count_df.index))
        with open("../src/british_spellings.json") as f:
            british_spell = json.load(f)
        count_df = count_df.rename(index=british_spell).rename(index=lemmas)
        count_df = count_df.groupby(count_df.index).sum()
        count_df = count_df[count_df.index.str.len() > 3]

        # df with each row corresponding to the frequency for one word
        # in each month (in how many papers does the word appear)
        counts_agg = count_df.to_numpy(copy=True)
        freqs = (counts_agg + 1) / (totals + 1)
        freqs_df = pd.DataFrame(dict(zip(months_w_years, list(freqs.T))), index=count_df.index)
        
        count_df.to_csv(
            os.path.join(results_path, sec, f"count_df.csv.gz"))
        freqs_df.to_csv(
            os.path.join(results_path, sec, f"freqs_df.csv.gz"))
        
    # save paper counts
    paper_count_dict["filters"] = {
        "article_type": article_type,
        "sections": sections,
        "full_text": full_text,
        "allow_other": allow_other,
        "token_pattern": token_pattern
    }
    paper_count_dict["lemmas"] = lemmas
    with open(os.path.join(results_path, "paper_counts.json"), "w") as f:
        json.dump(paper_count_dict, f)
    


def compute_frequency_projection(results_path: str):

    pred_range = 24  # make predictions based on 24 prior months
    end_date = "11-2022"  # last date to train reg with

    secs = next(os.walk(results_path))[1]
    for sec in secs:
        print(f"computing projection for section {sec}")
        freqs_df = pd.read_csv(
            os.path.join(results_path, sec, "freqs_df.csv.gz"),
            compression="gzip",
            index_col=0,
        )
        freqs_df = freqs_df.drop(
            ["7-2025", "8-2025", "9-2025", "10-2025", "11-2025", "12-2025"], axis=1
        )
        freqs_df = freqs_df[freqs_df.index.str.len() > 3]

        end_date_i = list(freqs_df.columns.values).index(end_date)
        start_date_i = end_date_i - pred_range
        y_i_all = freqs_df.columns.values[start_date_i:]
        y_i_pred = y_i_all[:pred_range]

        proj = np.zeros((freqs_df.shape[0], len(y_i_all)))

        for i, word in tqdm(enumerate(freqs_df.index)):
            y = freqs_df.loc[word]
            reg = LinearRegression().fit(np.arange(len(y_i_pred)).reshape(-1, 1), y[y_i_pred])
            proj[i, :] = reg.predict(np.arange(len(y_i_all)).reshape(-1, 1))
        
        freqs = freqs_df[y_i_all].to_numpy()
        ratios = freqs / proj
        diffs = freqs - proj

        np.save(os.path.join(results_path, sec, "proj.npy"), proj)
        np.save(os.path.join(results_path, sec, "ratios.npy"), ratios)
        np.save(os.path.join(results_path, sec, "diffs.npy"), diffs)


def load_freqs(data_path, words, start_date, end_date, x_months):

    secs = next(os.walk(data_path))[1]

    frequency_dfs = {}
    proj_dfs = {}
    diff_dfs = {}
    ratio_dfs = {}

    for sec in secs:
        df = pd.read_csv(
            os.path.join(data_path, sec, "freqs_df.csv.gz"),
            compression="gzip",
            index_col=0,
        )
        df = df[df.index.str.len() > 3]

        start_i = list(df.columns.values).index(start_date)
        end_i = list(df.columns.values).index(end_date) + 1
        frequency_dfs[sec] = df.loc[words][df.columns.values[start_i:end_i]]

        selection_mask = [x in words for x in df.index]
        selection_i = df.iloc[selection_mask].index

        proj = np.load(os.path.join(data_path, sec, "proj.npy"))
        proj_dfs[sec] = pd.DataFrame(
            proj[selection_mask, :], index=selection_i, columns=x_months
        )

        diff = np.load(os.path.join(data_path, sec, "diffs.npy"))
        diff_dfs[sec] = pd.DataFrame(
            diff[selection_mask, :], index=selection_i, columns=x_months
        )

        ratio = np.load(os.path.join(data_path, sec, "ratios.npy"))
        ratio_dfs[sec] = pd.DataFrame(
            ratio[selection_mask, :], index=selection_i, columns=x_months
        )
    
    frequency_dfs = restructure_by_words(frequency_dfs, words, secs, x_months)
    proj_dfs = restructure_by_words(proj_dfs, words, secs, x_months)
    diff_dfs = restructure_by_words(diff_dfs, words, secs, x_months)
    ratio_dfs = restructure_by_words(ratio_dfs, words, secs, x_months)

    for word in words:
        frequency_dfs[word]["projection"] = proj_dfs[word]["frequency"]
        frequency_dfs[word]["diff"] = diff_dfs[word]["frequency"]
        frequency_dfs[word]["ratio"] = ratio_dfs[word]["frequency"]
    
    return frequency_dfs

def restructure_by_words(df_dict, words, secs, x_months):
    word_dfs = {}
    for word in words:
        word_dict = {}
        for sec in secs:
            word_dict[sec] = df_dict[sec].loc[word].values
        df = pd.DataFrame.from_dict(word_dict, orient="index", columns=x_months)
        df = pd.melt(df, ignore_index=False).reset_index()
        df = df.rename(
            columns={"index": "section", "variable": "time", "value": "frequency"}
        )
        word_dfs[word] = df

    return word_dfs

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


chatgptwords_rare = [
    "accentuates", "acknowledges", "acknowledging", "addresses", "adept", "adhered", "adhering", "advancement", "advancements", "advancing", "advocates", "advocating", "affirming", "afflicted", "aiding", "akin", "align", "aligning", "aligns", "alongside", "amidst", "assessments", "attains", "attributed", "augmenting", "avenue", "avenues", "bolster", "bolstered", "bolstering", "broader", "burgeoning", "capabilities", "capitalizing", "categorized", "categorizes", "categorizing", "combating", "commendable", "compelling", "complicates", "complicating", "comprehending", "comprising", "consequently", "consolidates", "contributing", "conversely", "correlating", "crafted", "crafting", "culminating", "customizing", "delineates", "delve", "delved", "delves", "delving", "demonstrating", "dependability", "dependable", "detailing", "detrimentally", "diminishes", "diminishing", "discern", "discerned", "discernible", "discerning", "displaying", "disrupts", "distinctions", "distinctive", "elevate", "elevates", "elevating", "elucidate", "elucidates", "elucidating", "embracing", "emerges", "emphasises", "emphasising", "emphasize", "emphasizes", "emphasizing", "employing", "employs", "empowers", "emulating", "emulation", "enabling", "encapsulates", "encompass", "encompassed", "encompasses", "encompassing", "endeavors", "endeavours", "enduring", "enhancements", "enhances", "ensuring", "equipping", "escalating", "evaluates", "evolving", "exacerbating", "examines", "exceeding", "excels", "exceptional", "exceptionally", "exerting", "exhibiting", "exhibits", "expedite", "expediting", "exploration", "explores", "facilitated", "facilitates", "facilitating", "featuring", "formidable", "fostering", "fosters", "foundational", "furnish", "garnered", "garnering", "gauged", "grappling", "groundbreaking", "groundwork", "harness", "harnesses", "harnessing", "heighten", "heightened", "hinder", "hinges", "hinting", "hold", "holds", "illuminates", "illuminating", "imbalances", "impacting", "impede", "impeding", "imperative", "impressive", "inadequately", "incorporates", "incorporating", "influencing", "inherent", "initially", "innovative", "inquiries", "integrates", "integrating", "integration", "interconnectedness", "interplay", "intricacies", "intricate", "intricately", "introduces", "invaluable", "investigates", "involves", "juxtaposed", "leverages", "leveraging", "maintaining", "merges", "methodologies", "meticulous", "meticulously", "multifaceted", "necessitate", "necessitates", "necessitating", "necessity", "notable", "noteworthy", "nuanced", "nuances", "offering", "optimizing", "orchestrating", "outlines", "overlook", "overlooking", "paving", "persist", "pinpoint", "pinpointed", "pinpointing", "pioneering", "pioneers", "pivotal", "poised", "pose", "posed", "poses", "posing", "predominantly", "preserving", "pressing", "promise", "pronounced", "propelling", "realm", "realms", "recognizing", "refine", "refines", "refining", "remarkable", "renowned", "revealing", "reveals", "revolutionize", "revolutionizing", "revolves", "scrutinize", "scrutinized", "scrutinizing", "seamless", "seamlessly", "seeks", "serves", "serving", "shaping", "shedding", "showcased", "showcases", "showcasing", "signifying", "solidify", "spanned", "spanning", "spurred", "stands", "stemming", "strategically", "streamline", "streamlined", "streamlines", "streamlining", "struggle", "substantiated", "substantiates", "surged", "surmount", "surpass", "surpassed", "surpasses", "surpassing", "swift", "swiftly", "thorough", "transformative", "typically", "ultimately", "uncharted", "uncovering", "underexplored", "underscore", "underscored", "underscores", "underscoring", "unexplored", "unlocking", "unparalleled", "unraveling", "unveil", "unveiled", "unveiling", "unveils", "uphold", "upholding", "urging", "utilizes", "varying", "versatility", "warranting", "yielding"
]

chatgptwords_common = [
    "exhibited",
    "crucial",
    "additionally",
    "within",
    "notably",
    "insights",
    "comprehensive",
    "across",
    "particularly",
    "enhancing",
]