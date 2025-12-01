# llms_in_academia
current issues:
- different inflections as different words? (delve/delves/delved/delving)
 -> try to group them together

- when analyzing monthly frequency, which values should be used as the 
projection baseline? (Yearly averages?)
 -> use linear regression from the 24 months prior

- so far, monthly frequencies for word w are computed as 
(# papers with ocurrences of w / # papers in this month)
This ignores information about how often a word is used per paper
-> need new count vectorizer for that, think about that later



folder structure

results
|- baseline_2025-06-26
    |- selection criteria for papers used (article type, section names (maybe just first letter of each section to disambiguate) or all, if not filtered by section names)
        |- abstract
            |- counts
            |- count_df.csv
            |- freqs_df.csv
        |- methods
        |- results ...
        |- full (for vectorizing full text)
|- baseline_2025-12-26