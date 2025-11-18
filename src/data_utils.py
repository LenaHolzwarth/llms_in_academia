import re
import pandas as pd
from polyglot.detect import Detector
from polyglot.detect.base import Error

##### DATES #####

def get_alt_date(dates: list) -> list:
    '''take list of of unformatted dates and return list of pd.datetime objects
    this function expects the list entries to be in one of the following formats:
        - 17-11-2025
        - Nov-Dec-2025
        - Fall-2025
        - 17-11-2024-November-17
    '''

    dates_formatted = []
    for i, date in enumerate(dates):
        # if the date starts with letters, save them to get the alt_date entry
        r = r'^[a-zA-Z]*'
        key = re.match(r, date).group()
        # replace letters with correct digits
        if not key == '':
            r = r'.+(?=\-\d{4})'
            date = re.sub(r, alt_dates[key], date)
        # remove everything after year
        r = r'(?<=\-\d{4}).+'
        date = re.sub(r, '', date)
    
        try: 
            date = pd.to_datetime(date, dayfirst = True)
        except pd._libs.tslibs.parsing.DateParseError:
            print(f'DateParseError for {date} with index {i}')
        
        dates_formatted.append(date)
    
    return dates_formatted

# for the seasons, the meterological start of the season is used as the date 
# for time spans, the first date is used (although the span can be huge, e.g. 'Jan-Jun-2021' for PMC8297570)
alt_dates = {'Jan': '01-01',
             'January': '01-01',
             'Feb': '01-02',
             'February': '01-02',
             'Mar': '01-03',
             'March': '01-03',
             'Apr': '01-04',
             'April': '01-04',
             'May': '01-05',
             'Jun': '01-06',
             'June': '01-06',
             'Jul': '01-07',
             'July': '01-07',
             'Aug': '01-08',
             'August': '01-08',
             'Sep': '01-09',
             'September': '01-09',
             'Oct': '01-10',
             'October': '01-10',
             'Nov': '01-11',
             'November': '01-11',
             'Dec': '01-12',
             'December': '01-12',
             'Spring': '01-03',
             'Summer': '01-06',
             'Fall': '01-09',
             'Autumn': '01-09',
             'Winter': '01-12'}

##### LANGUAGE #####

def determine_lang(langs: list, texts: list):
    
    langs_new = []

    for i, lang in enumerate(langs):
        if lang == None and not texts[i] == None:
            try:
                lang = Detector(texts[i]).languages[0].code
            except Error:
                lang = None
        
        langs_new.append(lang)
    
    return langs_new

##### SECTIONS #####

def standardize_sections(sections: list, section_titles: list) -> list:
    """replaces list entries of 'sections' with dict entries, where each section is mapped to its standardized section name.
    sections without a standardized name are discarded
    if there is more than one section with the same standardized name (e.g. 'case 1' and 'case 2' would both be assigned to 
    'case report'), the section strings are concatenated 
    Arguments:
    - sections: nested list, each entry a list of strings, each is a section from a paper
    - section_titles: same as sections, but with corresponding title for each section
    """
    
    
    standardized = []
    
    for i, secs in enumerate(sections):
        sec_dict = {}
        if len(secs) == len(section_titles[i]):
            for j, sec in enumerate(secs):
                alt_title = get_alt_title(section_titles[i][j])
                if not alt_title == '':
                    if alt_title in sec_dict.keys():
                        sec_dict[alt_title] += ' ' + sec
                    else:
                        sec_dict[alt_title] = sec
        standardized.append(sec_dict)
        

    return standardized

##### SECTION TITLES #####

def get_alt_title(title: str) -> str:
    
    # sections we're not interested in
    ignore_list = ['supplementary', 'admin', 'combined', 'misc'] 
    
    # if title is not in the inclusion list, set to empty string
    alt = ''
    for key, synonyms in title_versions.items():
        if key in ignore_list:
            continue
        if not title == None:
            # clean title
            for pat, repl in title_clean.items():
                title = re.sub(pat, repl, title)
            title = title.strip(':').strip()
            if title.casefold() in synonyms:
                alt = key
    return alt



# strings to remove when processing section titles
title_clean = {r'\n': ' ', r'\ufeff': '', r' *\d+\.* *': '', r'\xa0': ' ', r'&': 'and'}


# standardized section titles
"""edge cases (need further investigation)
- theoretical framework
- implementation
- 'model'
- "recommendations"
- "outcomes",
- 'description' -> a combination of introduction and results, all from journal microPublication Biology
- 'diagnosis' -> results?
- 'analysis' / 'data analysis' / "statistical analyses", -> sometimes methods, sometimes results
- 'summary' is sometimes in place of abstract, sometimes as part of conclusion
- 'highlights'
"""
title_versions = {
    "introduction": [
        "aim",
        "introduction",
        "introduction and background",
        "introduction and importance" "main",
        "objective",
        "objectives",
        "⧉ introduction",
    ],
    "background": [
        "background",
        "literature review",
        "preliminaries",
        "related literature",
        "related work",
        "related works",
        "review",
        "theoretical background",
        "theory",
    ],
    "methods": [
        "computational details",
        "data",
        "data and methodologies",
        "data and methods",
        "data collection",
        "data description",
        "data records",
        "experimental design, materials and methods",
        "experimental methods",
        "material and method",
        "material and methods",
        "materials",
        "materials and equipment",
        "materials and method",
        "materials and methods",
        "measures",
        "method",
        "method details",
        "methodology",
        "methods",
        "methods/design",
        "methods and design",
        "methods and materials",
        "participants and methods",
        "patients and methods",
        "procedure",
        "research design",
        "research design and methods" "research methodology",
        "research methods and design",
        "star★methods",
        "statistical analysis",
        "subjects and methods",
        "technical validation",
    ],
    "case report": [
        "case",
        "case history",
        "case description",
        "case discussion",
        "case presentation",
        "case presentations",
        "case report",
        "case report/case presentation",
        "case reports",
        "case series",
        "case study",
        "cases presentation",
        "clinical case",
        "clinical history",
        "illustrative case",
        "past medical history",
        "patient and observation",
        "presentation of case",
    ],
    "outcome": ["outcome", "outcomes", "outcome and follow-up"],
    "experiment": [
        "experiment",
        "experiment 1",
        "experiment 2",
        "experimental",
        "experimental details",
        "experimental procedure",
        "experimental procedures",
        "experimental section",
        "experimental setup",
        "experiments",
    ],
    "treatment": ["treatment", "treatment plan", "treatment progress"],
    "diagnosis": [
        "diagnoses",
        "diagnosis",
        "final diagnosis",
        "differential diagnosis",
    ],
    "results": [
        "experimental results",
        "experimental results and analysis",
        "findings",
        "main results",
        "result",
        "results",
        "results and analysis",
        "⧉ results",
    ],
    "discussion": [
        "discussion",
        "discussions",
        "general discussion",
        "utility and discussion",
        "⧉ discussions",
    ],
    "conclusion": [
        "challenges and future directions",
        "clinical implications",
        "concluding remarks",
        "concluding remarks and future directions",
        "concluding remarks and future perspectives",
        "conclusion",
        "conclusion and future directions",
        "conclusion and future perspective",
        "conclusion and future perspectives",
        "conclusion and future work",
        "conclusion and outlook",
        "conclusion and perspective" "conclusion and perspectives",
        "conclusion and recommendations",
        "conclusions",
        "conclusions/outlook",
        "conclusions and future directions",
        "conclusions and future perspectives",
        "conclusions and future work",
        "conclusions and perspective",
        "conclusions and perspectives",
        "conclusions and prospects",
        "conclusions and outlook",
        "conclusions and recommendations",
        "final considerations",
        "future direction",
        "future directions",
        "future perspective",
        "future perspectives",
        "future research directions",
        "implications",
        "outlook",
        "perspectives",
        "summary and conclusion",
        "summary and conclusions",
        "summary and future perspectives" "summary and outlook",
        "summary and perspectives",
        "⧉ conclusions",
    ],
    "limitations": [
        "limitation",
        "limitation of the study",
        "limitations",
        "limitations of the study",
        "strengths and limitations",
        "study limitations",
    ],
    "summary": ["summary", "key messages", "key summary points", "key points"],
    "supplementary": [
        "supporting information",
        "supplementary information",
        "supplementary material",
        "electronic supplementary material",
        "supplemental material",
        "supplemental information",
        "online content",
        "abbreviations",
        "source data",
        "supplementary materials",
        "patents",
        "value of the data",
        "additional information",
        "supplementary data",
        "supplementary figures and tables",
        "trial status",
        "additional file",
        "taxonomy",
        "appendix",
        "supplementary material figures",
    ],
    "admin": [
        "author contributions",
        "conflicts of interest",
        "ethics statement",
        "conflict of interest",
        "conflict of interest statement",
        "declaration of competing interest",
        "credit authorship contribution statement",
        "funding",
        "consent",
        "data availability",
        "acknowledgments",
        "funding information",
        "declaration of interests",
        "conflict of interests",
        "disclosure",
        "ethical approval",
        "resource availability",
        "disclosures",
        "availability of data and materials",
        "competing interests",
        "declaration of generative ai and ai-assisted technologies in the writing process",
        "authors’ contributions",
        "consent for publication",
        "ethics approval and consent to participate",
        "contributors",
        "data availability statement",
        "peer review",
        "transparency statement",
        "data sharing statement",
        "ethics approval",
        "author contribution",
        "announcement",
        "declarations",
        "credit author statement",
        "disclaimer",
        "patient consent",
        "consent statement",
        "authors' contributions",
        "authors’ information",
        "availability of supporting data",
        "pre-publication history",
        "competing interest",
        "authors’ contribution",
        "additional files",
        "supplementary figures",
        "availability and requirements",
        "sources of funding",
        "acknowledgements",
    ],
    "combined": [
        "results and discussion",
        "discussion and conclusion",
        "result and discussion",
        "limitations and future directions",
        "background and summary",
        "methods and analysis",
        "experiments and results",
        "conclusion and discussion",
        "experimental results and discussion",
        "limitations and future research",
        "results and discussions",
        "discussion and conclusions",
    ],
    "misc": [
        "",
        "introdução",
        "resultados",
        "discussão",
        "chemical context",
        "synthesis and crystallization",
        "structural commentary",
        "supra\xadmolecular features",
        "refinement",
        "implementation",
    ],
}
