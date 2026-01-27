import os
import json
import random
import pandas as pd
import numpy as np
import scipy as sp
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LinearRegression


def load_paper_df(
    data_path: str,
    article_type: str = "all",
    sections: list = [],
    full_text: bool = True,
    allow_other: bool = False,
    subset: str = "all",
):
    """
    load and filter data frame with pubmed papers

    Args:
        data_path: where the paper df is stored
        article_type: should be one of ["all", "research-article",
                      "review-article", "case-report"]
        sections: which sections should papers have.
        full_text: should the full text be vectorized as one?
                   (additionally to each section being vectorized on its own)
        allow_other: do sections need to be a direct match (i.e., should
                     papers that have more than the required sections
                     also be disregarded)
        subset: load all papers ("all"), or just one specific year (e.g. "2024")
    """

    if subset == "all":
        df = pd.read_pickle(os.path.join(data_path, f"{subset}.pkl"))
    else:
        df = pd.read_json(os.path.join(data_path, f"{subset}.json"))

    paper_count_full = len(df)

    if not article_type == "all":
        df = df[df["article-type"] == article_type]

    # reformat data frame to only include the sections that are required
    if not sections == []:
        if allow_other:
            df = df[df["sections"].apply(lambda x: set(x.keys()) >= set(sections))]
        else:
            df = df[df["sections"].apply(lambda x: set(x.keys()) == set(sections))]

        for sec in sections:
            df[sec] = list(map((lambda x: x[sec]), df["sections"]))
            # only include papers where each section has length >= 250
            df = df[df[sec].apply(lambda x: len(x) >= 250)]

        # also filter abstract length
        df = df[df["abstract"].apply(lambda x: len(x) >= 250)]

        if full_text:
            df["full"] = [""] * len(df)
            for sec in sections:
                df["full"] = df["full"] + df[sec]

        paper_count_filtered = len(df)

    return df, paper_count_full, paper_count_filtered


def compute_word_frequency(
    data_path: str,
    results_path: str,
    article_type: str = "all",
    sections: list = [],
    full_text: bool = True,
    allow_other: bool = False,
    token_pattern: str = r"(?u)\b[A-Za-z]{4,}\b",
    min_token_len: int = 4,
    version: str = "",
    max_len: int = -1,
):
    """
    Get monthly word frequency for each section of PMC papers.
    Process:
    - select papers with matching sections
    - run every section through binary count vectorizer
    - aggregate counts from individual paper to monthly
    - use monthly counts to compute frequency
    - lemmatize: aggregate frequencies of different versions of the same word

    Args:
        data_path: where the paper df is stored
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
        min_token_len: don't tokenize words shorten than this, make sure this
                        is in line with the token_pattern!
        version: one of ["crop", "sample", "] determine if the texts
                 should be cropped or sampled to be at most as long as max_len
        max_len: texts should not have more words than this. Fewer is okay
    """
    years = np.arange(2000, 2026)
    months = np.arange(1, 13)

    os.makedirs(os.path.join(results_path, "abstract"), exist_ok=False)
    if full_text:
        os.makedirs(os.path.join(results_path, "full"), exist_ok=False)
    for sec in sections:
        os.makedirs(os.path.join(results_path, sec), exist_ok=False)

    df, paper_count_full, paper_count_filtered = load_paper_df(
        data_path, article_type, sections, full_text, allow_other
    )
    filters_dict = {
        "papers_before_filters": paper_count_full,
        "papers_after_filters": paper_count_filtered,
    }

    filters_dict["all_dates"] = list(map(str, df["date"].values))

    sections.append("abstract")
    if full_text:
        sections.append("full")

    for sec in sections:
        print(f"vectorizing section {sec}")
        vectorizer = CountVectorizer(
            binary=True, min_df=1e-6, token_pattern=token_pattern
        )

        random.seed(0)
        if version == "crop":
            cropped = list(map(lambda x: crop_section(x, max_len), df[sec]))
            X = vectorizer.fit_transform(cropped)
        elif version == "sample":
            sampled = list(map(lambda x: sample_section(x, max_len), df[sec]))
            X = vectorizer.fit_transform(sampled)
        else:
            X = vectorizer.fit_transform(df[sec].values)

        sp.sparse.save_npz(os.path.join(results_path, sec, f"count_{sec}.pkl"), X)
        print("vectorizing complete")

        words = vectorizer.get_feature_names_out()
        #### need to save words at this point!
        np.save(
            os.path.join(results_path, sec, f"words_{sec}.pkl"),
            words,
            allow_pickle=True,
        )

        counts = np.zeros((words.size, 12 * years.size))
        totals = np.zeros(12 * years.size)

        for j, year in enumerate(years):
            if year == 2025:
                months = np.arange(1, 7)

            for i, month in enumerate(months):
                ind = (
                    df["date"]
                    .apply(lambda x: (x.month == month) and (x.year == year))
                    .values
                )
                # count how many times a word appears in each month
                counts[:, 12 * j + i] = np.array(np.sum(X[ind, :], axis=0)).ravel()
                # count papers per month
                totals[12 * j + i] = np.sum(ind)

            if year == 2025:
                months = np.arange(1, 13)  # set it back for the other sections!

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
        # lemmatizing shortens some strings below threshold, they should be removed
        count_df = count_df[count_df.index.str.len() > min_token_len]
        # save lemmas for each section
        with open(os.path.join(results_path, sec, "lemmas.json"), "w") as f:
            json.dump(lemmas, f)
        # also save lemmatized word list?
        np.save(
            os.path.join(results_path, sec, "words_lemmatized.pkl"),
            count_df.index,
            allow_pickle=True,
        )

        # df with each row corresponding to the frequency for one word
        # in each month (in how many papers does the word appear)
        counts_agg = count_df.to_numpy(copy=True)
        freqs = (counts_agg + 1) / (totals + 1)
        freqs_df = pd.DataFrame(
            dict(zip(months_w_years, list(freqs.T))), index=count_df.index
        )

        count_df.to_csv(os.path.join(results_path, sec, f"count_df.csv.gz"))
        freqs_df.to_csv(os.path.join(results_path, sec, f"freqs_df.csv.gz"))

    # save paper counts
    filters_dict["filters"] = {
        "article_type": article_type,
        "sections": sections,
        "full_text": full_text,
        "allow_other": allow_other,
        "token_pattern": token_pattern,
        "version": version,
        "max_len": max_len,
    }

    with open(os.path.join(results_path, "filters.json"), "w") as f:
        json.dump(filters_dict, f)


def compute_group_frequency(
    data_path: str,
    results_path: str,
):
    """
    Docstring for compute_group_frequency

    Args:
        data_path: where the paper df is stored
        results_path: path to results/baseline_name/filters
    """
    years = np.arange(2000, 2026)
    months = np.arange(1, 13)

    with open(os.path.join(results_path, "filters.json")) as f:
        filters = json.load(f)

    article_type = filters["filters"]["article_type"]
    sections = filters["filters"]["sections"]
    print(f"sections: {sections}")
    sections = [sec for sec in sections if not sec in ["abstract", "full"]]
    full_text = filters["filters"]["full_text"]
    allow_other = filters["filters"]["allow_other"]

    print(f"article type: {article_type}")
    print(f"sections: {sections}")
    print(f"full_text: {full_text}")
    print(f"allow_other: {allow_other}")

    # get dates -> replace this with getting the dates from filters.json
    """
    dates, _, _ = load_paper_df(
        data_path, article_type, sections, full_text, allow_other
    )
    dates = dates["date"]
    """
    dates = pd.to_datetime(filters["all_dates"])
    print(f"dates shape: {dates.shape}")

    sections.append("abstract")
    if full_text:
        sections.append("full")

    for sec in sections:
        print(f"counting section {sec}")
        X = sp.sparse.load_npz(os.path.join(results_path, sec, f"count_{sec}.pkl.npz"))
        words = np.load(
            os.path.join(results_path, sec, f"words_{sec}.pkl.npy"), allow_pickle=True
        )

        ind_words_common = np.isin(words, chatgptwords_common)
        ind_words_rare = np.isin(words, chatgptwords_rare)

        group_counts = np.zeros((2, 12 * years.size))
        group_totals = np.zeros(12 * years.size)

        for j, year in enumerate(years):
            if year == 2025:
                months = np.arange(1, 7)

            for i, month in enumerate(months):
                # this might need to be changed with the new date format
                # ind = (dates.dt.month == month) & (dates.dt.year == year)
                # ind = ind.to_numpy(dtype=bool)
                # with new date format
                ind = (dates.month == month) & (dates.year == year)
                # print(f"ind shape: {ind.shape}")
                # print(f"X shape: {X.shape}")
                for k, ind_words in enumerate([ind_words_common, ind_words_rare]):
                    # count how many times any word from the selection appears in each month
                    group_counts[k, 12 * j + i] = np.sum(
                        np.sum(X[ind, :][:, ind_words], axis=1) > 0
                    )
                    # count papers per month
                    group_totals[12 * j + i] = np.sum(ind)

            if year == 2025:
                months = np.arange(1, 13)  # set it back for the other sections!

        months_w_years = [f"{m}-{y}" for y in years for m in months]
        group_count_df = pd.DataFrame(
            dict(zip(months_w_years, list(group_counts.astype(int).T))),
            index=["common_words", "rare_words"],
        )

        counts_agg = group_count_df.to_numpy(copy=True)
        freqs = (counts_agg + 1) / (group_totals + 1)
        group_freqs_df = pd.DataFrame(
            dict(zip(months_w_years, list(freqs.T))), index=group_count_df.index
        )

        group_count_df.to_csv(os.path.join(results_path, sec, f"group_count_df.csv.gz"))
        group_freqs_df.to_csv(os.path.join(results_path, sec, f"group_freqs_df.csv.gz"))


def compute_frequency_projection(
    results_path: str,
    pred_range: int = 24,
    end_date: str = "11-2022",
    group_prefix: str = "",
    cutoff: float = 1e-3,
):
    """
    uses linear regression to predict trends in monthly word frequency.
    The time covered by the prediction ranges from (end_date - pred_range)
    to the last time-point in the observed frequencies
    It saves the projection, as well as the difference (obs - pred) and
    ratio (obs / proj) between observed and predicted frequency

    ISSUE: which month to use to evaluate cutoff? last recorded? mean over 12
           last recorded months?


    Args:
        results_path: directory with word frequency dataframes
        pred_range: number of months to fit the regression on
        end_date: the month before end_date is the last month used in regression
        group_prefix: one of "" (projections for individual words),
                      "group_" (projections for word groups)
        cutoff: ignore words with lower frequency than cutoff
    """

    secs = next(os.walk(results_path))[1]
    for sec in secs:
        print(f"computing projection for section {sec}")
        freqs_df = pd.read_csv(
            os.path.join(results_path, sec, f"{group_prefix}freqs_df.csv.gz"),
            compression="gzip",
            index_col=0,
        )
        ###### DELETE FOR NEW BASELINE #######
        freqs_df = freqs_df.drop(
            ["7-2025", "8-2025", "9-2025", "10-2025", "11-2025", "12-2025"], axis=1
        )
        # freqs_df = freqs_df[freqs_df.index.str.len() > 3]

        end_date_i = list(freqs_df.columns.values).index(end_date)
        start_date_i = end_date_i - pred_range
        y_i_all = freqs_df.columns.values[start_date_i:]
        y_i_pred = y_i_all[:pred_range]

        # cutoff based on the last month recorded
        freqs_df = freqs_df[freqs_df.iloc[:, -1] >= cutoff]
        np.save(
            os.path.join(results_path, sec, "words_after_cutoff.pkl"),
            freqs_df.index,
            allow_pickle=True,
        )

        proj = np.zeros((freqs_df.shape[0], len(y_i_all)))

        for i, word in tqdm(enumerate(freqs_df.index)):
            y = freqs_df.loc[word]
            reg = LinearRegression().fit(
                np.arange(len(y_i_pred)).reshape(-1, 1), y[y_i_pred]
            )
            proj[i, :] = reg.predict(np.arange(len(y_i_all)).reshape(-1, 1))

        freqs = freqs_df[y_i_all].to_numpy()
        ratios = freqs / proj
        diffs = freqs - proj

        np.save(os.path.join(results_path, sec, f"{group_prefix}freqs.npy"), freqs)
        np.save(os.path.join(results_path, sec, f"{group_prefix}proj.npy"), proj)
        np.save(os.path.join(results_path, sec, f"{group_prefix}ratios.npy"), ratios)
        np.save(os.path.join(results_path, sec, f"{group_prefix}diffs.npy"), diffs)


def get_frequency_projection_yearly(results_path: str, sec: str, cutoff: float = 1e-3):
    """
    predict the mean frequency for every word and year for one section.
    Frequency is predicted via interpolation from the mean frequencies
    two and three years prior. Returns the actual frequencies, diff and
    ratios

    Args:
        results_path: directory with word frequency dataframes
        cutoff: ignore words with lower frequency than cutoff
    """

    results_path = os.path.join(results_path, sec)
    freqs_path = os.path.join(results_path, "yearly_freqs_df.csv.gz")

    if os.path.exists(freqs_path):
        print("loading yearly projection")
        yearly_freqs = pd.read_csv(
            freqs_path,
            compression="gzip",
            index_col=0,
        )
        yearly_diffs = pd.read_csv(
            os.path.join(results_path, "yearly_diffs_df.csv.gz"),
            compression="gzip",
            index_col=0,
        )
        yearly_ratios = pd.read_csv(
            os.path.join(results_path, "yearly_ratios_df.csv.gz"),
            compression="gzip",
            index_col=0,
        )

    else:
        print("computing yearly projection")
        freqs_df = pd.read_csv(
            os.path.join(results_path, "freqs_df.csv.gz"),
            compression="gzip",
            index_col=0,
        )
        freqs_df = freqs_df.drop(
            ["7-2025", "8-2025", "9-2025", "10-2025", "11-2025", "12-2025"], axis=1
        )
        # remove frequencies below cutoff
        #freqs_df = freqs_df[freqs_df.iloc[:, -1] >= cutoff]
        #freqs_df = freqs_df[freqs_df >= cutoff]
        words = freqs_df.index

        n_months = 12
        yearly_freqs = {}
        for year in range(2010, 2026):
            if year == 2025:
                n_months = 6

            ind = np.where(freqs_df.columns.values == f"1-{year}")[0][0]
            current_freqs = freqs_df.iloc[:, ind : ind + n_months]
            yearly_freqs[str(year)] = current_freqs.mean(axis=1)

        yearly_freqs = pd.DataFrame(yearly_freqs, index=words)
        yearly_projection = np.zeros(yearly_freqs.shape)[:, :-3]
        for i, year in enumerate(range(2013, 2026)):
            yearly_projection[:, i] = yearly_freqs[str(year - 2)] + np.maximum(
                (yearly_freqs[str(year - 2)] - yearly_freqs[str(year - 3)]) * 2, 0
            )

        yearly_diffs = yearly_freqs.iloc[:, 3:] - yearly_projection
        yearly_ratios = yearly_freqs.iloc[:, 3:] / yearly_projection

        yearly_freqs.to_csv(os.path.join(freqs_path))
        yearly_diffs.to_csv(
            os.path.join(results_path, "yearly_diffs_df.csv.gz")
        )
        yearly_ratios.to_csv(
            os.path.join(results_path, "yearly_ratios_df.csv.gz")
        )

    return yearly_freqs, yearly_diffs, yearly_ratios


def load_freqs(
    data_path: str,
    words: list = ["common_words", "rare_words"],
    start_date: str = "11-2020",
    end_date: str = "6-2025",
    group_prefix: str = "",
):
    """
    Get the frequencies for specific words.
    Returns a dictionary that maps a frequency dataframe to each word in words.
    The dataframe contains the observed and predicted frequency, as well as
    their difference (obs - pred) and ratio (obs / pred) and has columns
    "section", "time", "frequency", "projection", "diff", "ratio"

    ISSUE:  what to do when the queried time span is not identical to the
            time span covered by the prediction?
            to solve, need to know the time span of the prediction,
            because all predictions go to the latest date (6-2025),
            the start date can be inferred by subtraction

    Args:
        data_path: directory with word frequency dataframes
        words: selection of words whose frequencies are to be returned
                (default value is the group names)
        start_date: return frequencies starting from this month
        end_date: return frequencies up to this month
        group_prefix: one of "" (projections for individual words),
                      "group_" (projections for word groups)
    """
    if not (start_date == "11-2020" and end_date == "6-2025"):
        raise Exception(
            "functionality for time span outside of prediction range not yet implemented"
        )

    # build continuous time axis for plotting (i.e. 2020.83 corresponds to nov-2020)
    start_split = start_date.split("-")
    end_split = end_date.split("-")
    x_months = np.arange(
        int(start_split[1]) + ((int(start_split[0]) - 1) / 12),
        int(end_split[1]) + ((int(end_split[0]) - 1) / 12),
        1 / 12,
    )

    secs = next(os.walk(data_path))[1]

    frequency_dfs = {}
    proj_dfs = {}
    diff_dfs = {}
    ratio_dfs = {}

    for sec in secs:
        df = pd.read_csv(
            os.path.join(data_path, sec, f"{group_prefix}freqs_df.csv.gz"),
            compression="gzip",
            index_col=0,
        )
        df = df[df.index.str.len() > 3]

        start_i = list(df.columns.values).index(start_date)
        end_i = list(df.columns.values).index(end_date) + 1
        frequency_dfs[sec] = df.loc[words][df.columns.values[start_i:end_i]]

        selection_mask = [x in words for x in df.index]
        selection_i = df.iloc[selection_mask].index

        proj = np.load(os.path.join(data_path, sec, f"{group_prefix}proj.npy"))
        # if the time span given here doesn't line up with the prediction span,
        # here is the place to adjust the shape of proj to match x_months
        proj_dfs[sec] = pd.DataFrame(
            proj[selection_mask, :], index=selection_i, columns=x_months
        )

        diff = np.load(os.path.join(data_path, sec, f"{group_prefix}diffs.npy"))
        diff_dfs[sec] = pd.DataFrame(
            diff[selection_mask, :], index=selection_i, columns=x_months
        )

        ratio = np.load(os.path.join(data_path, sec, f"{group_prefix}ratios.npy"))
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


def crop_section(sec: str, max_len: int):
    """returns a section of sec with a given number of words. If sec is
    shorter than the required word count, return sec unchanged.
    The section of words that is returned is from a random place in sec.
    IMPORTANT: The random seed needs to be set outside of this function
    """
    sec_words = sec.split()
    sec_len = len(sec_words)

    if sec_len <= max_len:
        return sec

    else:
        start_i = random.randint(0, sec_len - max_len)
        end_i = start_i + max_len
        return " ".join(sec_words[start_i:end_i])


def sample_section(sec: str, max_len: int):
    """returns randomly sampled words (without replacement) from sec
    If sec is shorter than the required word count, return sec unchanged.
    IMPORTANT: The random seed needs to be set outside of this function
    """
    sec_words = sec.split()

    if len(sec_words) <= max_len:
        return sec

    else:
        return " ".join(random.sample(sec_words, max_len))


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

chatgptwords_rare = [
    "accentuates",
    "acknowledges",
    "acknowledging",
    "addresses",
    "adept",
    "adhered",
    "adhering",
    "advancement",
    "advancements",
    "advancing",
    "advocates",
    "advocating",
    "affirming",
    "afflicted",
    "aiding",
    "akin",
    "align",
    "aligning",
    "aligns",
    "alongside",
    "amidst",
    "assessments",
    "attains",
    "attributed",
    "augmenting",
    "avenue",
    "avenues",
    "bolster",
    "bolstered",
    "bolstering",
    "broader",
    "burgeoning",
    "capabilities",
    "capitalizing",
    "categorized",
    "categorizes",
    "categorizing",
    "combating",
    "commendable",
    "compelling",
    "complicates",
    "complicating",
    "comprehending",
    "comprising",
    "consequently",
    "consolidates",
    "contributing",
    "conversely",
    "correlating",
    "crafted",
    "crafting",
    "culminating",
    "customizing",
    "delineates",
    "delve",
    "delved",
    "delves",
    "delving",
    "demonstrating",
    "dependability",
    "dependable",
    "detailing",
    "detrimentally",
    "diminishes",
    "diminishing",
    "discern",
    "discerned",
    "discernible",
    "discerning",
    "displaying",
    "disrupts",
    "distinctions",
    "distinctive",
    "elevate",
    "elevates",
    "elevating",
    "elucidate",
    "elucidates",
    "elucidating",
    "embracing",
    "emerges",
    "emphasises",
    "emphasising",
    "emphasize",
    "emphasizes",
    "emphasizing",
    "employing",
    "employs",
    "empowers",
    "emulating",
    "emulation",
    "enabling",
    "encapsulates",
    "encompass",
    "encompassed",
    "encompasses",
    "encompassing",
    "endeavors",
    "endeavours",
    "enduring",
    "enhancements",
    "enhances",
    "ensuring",
    "equipping",
    "escalating",
    "evaluates",
    "evolving",
    "exacerbating",
    "examines",
    "exceeding",
    "excels",
    "exceptional",
    "exceptionally",
    "exerting",
    "exhibiting",
    "exhibits",
    "expedite",
    "expediting",
    "exploration",
    "explores",
    "facilitated",
    "facilitates",
    "facilitating",
    "featuring",
    "formidable",
    "fostering",
    "fosters",
    "foundational",
    "furnish",
    "garnered",
    "garnering",
    "gauged",
    "grappling",
    "groundbreaking",
    "groundwork",
    "harness",
    "harnesses",
    "harnessing",
    "heighten",
    "heightened",
    "hinder",
    "hinges",
    "hinting",
    "hold",
    "holds",
    "illuminates",
    "illuminating",
    "imbalances",
    "impacting",
    "impede",
    "impeding",
    "imperative",
    "impressive",
    "inadequately",
    "incorporates",
    "incorporating",
    "influencing",
    "inherent",
    "initially",
    "innovative",
    "inquiries",
    "integrates",
    "integrating",
    "integration",
    "interconnectedness",
    "interplay",
    "intricacies",
    "intricate",
    "intricately",
    "introduces",
    "invaluable",
    "investigates",
    "involves",
    "juxtaposed",
    "leverages",
    "leveraging",
    "maintaining",
    "merges",
    "methodologies",
    "meticulous",
    "meticulously",
    "multifaceted",
    "necessitate",
    "necessitates",
    "necessitating",
    "necessity",
    "notable",
    "noteworthy",
    "nuanced",
    "nuances",
    "offering",
    "optimizing",
    "orchestrating",
    "outlines",
    "overlook",
    "overlooking",
    "paving",
    "persist",
    "pinpoint",
    "pinpointed",
    "pinpointing",
    "pioneering",
    "pioneers",
    "pivotal",
    "poised",
    "pose",
    "posed",
    "poses",
    "posing",
    "predominantly",
    "preserving",
    "pressing",
    "promise",
    "pronounced",
    "propelling",
    "realm",
    "realms",
    "recognizing",
    "refine",
    "refines",
    "refining",
    "remarkable",
    "renowned",
    "revealing",
    "reveals",
    "revolutionize",
    "revolutionizing",
    "revolves",
    "scrutinize",
    "scrutinized",
    "scrutinizing",
    "seamless",
    "seamlessly",
    "seeks",
    "serves",
    "serving",
    "shaping",
    "shedding",
    "showcased",
    "showcases",
    "showcasing",
    "signifying",
    "solidify",
    "spanned",
    "spanning",
    "spurred",
    "stands",
    "stemming",
    "strategically",
    "streamline",
    "streamlined",
    "streamlines",
    "streamlining",
    "struggle",
    "substantiated",
    "substantiates",
    "surged",
    "surmount",
    "surpass",
    "surpassed",
    "surpasses",
    "surpassing",
    "swift",
    "swiftly",
    "thorough",
    "transformative",
    "typically",
    "ultimately",
    "uncharted",
    "uncovering",
    "underexplored",
    "underscore",
    "underscored",
    "underscores",
    "underscoring",
    "unexplored",
    "unlocking",
    "unparalleled",
    "unraveling",
    "unveil",
    "unveiled",
    "unveiling",
    "unveils",
    "uphold",
    "upholding",
    "urging",
    "utilizes",
    "varying",
    "versatility",
    "warranting",
    "yielding",
]

ignore_list = [
    "abstract",
    "abstractgraphical",
    "actabiomedica",
    "amsbsy",
    "amsfonts",
    "amsmath",
    "amssymb", 
    "article",
    "background",
    "bstract",
    "clinicaltrial",
    "commentary",
    "conclusion",
    "discussion",
    "documentclass",
    "download",
    "editor",
    "elife",
    "github",
    "grant",
    "introduction",
    "mathrsfs",
    "method",
    "objective",
    "oddsidemargin",
    "pubmed",
    "result",
    "scholar",
    "setlength",
    "showproj",
    "signifi",  # significant sometimes has a whitespace inside?
    "supplemental",
    "supplementary",
    "textsupplemental",
    "tinyurl",
    "upgreek",
    "usepackage",
    "wasysym",
    "wiley",
    
]

geo_list = [
    "arabia",
    "barcelona",
    "daegu",
    "fudan",
    "ghana",
    "henan",
    "hubei",
    "israel",
    "italy",
    "jeddah",
    "joanna", #Joanna Briggs Institute
    "jordanian",
    "kyoto",
    "lebanon",
    "liberia",
    "lombardy",
    "macedonia",
    "oxford",
    "peking",
    "pittsburgh",
    "qinghai",
    "riyadh",
    "saudi",
    "syrian",
    "taipei",
    "tibetan",
    "tongji",
    "wuhan",
]

uncertain_list = [
    # names of people and places, months
    "keywords",
    "objective",
    "online",  # "materials available online", "online course"
    "please",  # as in "please refer to", but also "patients were pleased"
]
