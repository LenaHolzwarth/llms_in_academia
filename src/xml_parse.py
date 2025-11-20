"""
Module to parse the pmc .xml files
Each file corresponds to one paper and contains meta information and the 
text itself
"""

import xml.etree.ElementTree as et
import pandas as pd
import re
import os
from tqdm import tqdm
from iso3166 import countries


def xml_parse_single(xml_file: str) -> dict:
    """
    Parses a single .xml file and stores the information in a dict.
    The dict keys need to match those specified in 
    xml_parse_baseline_segment

    Args:
      xml_file (str): path and name of file to be parsed

    Returns:
      dict: dictionary containing the information parsed from the file
    """

    paper_dict = {}

    xtree = et.parse(xml_file, parser=et.XMLParser(encoding="UTF-8"))
    xroot = xtree.getroot()

    paper_dict["article-type"] = xml_get_attr(xroot, "article-type")
    paper_dict["language"] = xml_get_attr(
        xroot, "{http://www.w3.org/XML/1998/namespace}lang"
    )
    paper_dict["journal"] = xml_get_text(
        xroot.findall("./front/journal-meta/journal-title-group/journal-title")
    )
    paper_dict["pmc-id"] = xml_get_text(
        xroot.findall('./front/article-meta/article-id/[@pub-id-type="pmc"]')
    )
    paper_dict["pmid"] = xml_get_text(
        xroot.findall('./front/article-meta/article-id/[@pub-id-type="pmid"]')
    )
    paper_dict["title"] = xml_get_text(
        xroot.findall("./front/article-meta/title-group/article-title")
    )
    paper_dict["country"] = get_country(xroot)
    paper_dict["date"] = get_date(xroot)
    paper_dict["abstract"] = get_abstr(
        xroot.findall("./front/article-meta/abstract")
    )
    sections_dict = get_sections(xroot)
    paper_dict["section_titles"] = list(sections_dict.keys())
    paper_dict["sections"] = list(sections_dict.values())

    return paper_dict


def xml_parse_baseline_segment(data_path: str, json_path: str) -> pd.DataFrame:
    """
    Parse all of the .xml files in a folder into one pandas dataframe, 
    which is saved to a json file.
    Args:
      - data_path (str): location of folder
      - json_path (str): path and name of the output json file
    """
    # set up dict to collect individiual papers
    keys = [
        "article-type",
        "language",
        "journal",
        "pmc-id",
        "pmid",
        "title",
        "country",
        "date",
        "abstract",
        "section_titles",
        "sections",
    ]
    baseline = {k: [] for k in keys}

    for dirpath, _, filenames in os.walk(data_path):
        for file in tqdm(filenames):
            # check that only files named 'PMCxxxxxxxx.xml' are being processed
            r = re.compile("PMC\\d{6,8}.xml")
            if r.match(file):
                try:
                    paper_dict = xml_parse_single(os.path.join(dirpath, file))
                except et.ParseError as e:
                    print(f"parse error in file {file}: {e}")
                else:
                    # append new paper to baseline dict
                    {k: v.append(paper_dict[k]) for k, v in baseline.items()}

    baseline_df = pd.DataFrame.from_dict(baseline)
    baseline_df.to_json(json_path)

    return baseline_df


def xml_parse_baseline(
    data_path: str, json_path: str = "../data", replace: bool = False
):
    """
    Parse multiple folders of .xml files and save the extracted info in one 
    json file per folder.

    Args:
      data_path (str): location of .xml files
      json_path (str): location of .json output
      replace (bool): if a .json file of same name already exists, 
                      should it be replaced?
    """

    dirname = os.path.basename(data_path)
    subdirs = next(os.walk(data_path))[1]

    for subdir in subdirs:
        subdir_path = os.path.join(data_path, subdir)
        json_file = os.path.join(json_path, dirname, subdir + ".json")

        print(f"parsing {subdir}")
        if replace or not os.path.exists(json_file):
            _ = xml_parse_baseline_segment(subdir_path, json_file)


### HELPER FUNCTIONS ###
def xml_get_attr(node, attr_name: str):
    if attr_name in node.attrib.keys():
        return node.attrib[attr_name]
    else:
        return None


def xml_get_text(nodes: list, joinstr: str = " ") -> str:
    """
    Function to get the text from xml elements. It expects a list of
    elements and will only return the text from the first list item.

    Args:
      nodes (list): list of xml elements to be parsed
      joinstr (str): string used to concatenate the text found in the 
                     children of the first xml element
    
    Returns:
      str | None: text from the first element in nodes (or None)
    """
    if len(nodes) > 0:
        return joinstr.join(nodes[0].itertext())  # .text
    else:
        return None


def get_abstr(node):
    abs = ""
    for a in node:
        if "graphical" in a.attrib.values():
            continue
        abs += xml_get_text(node)

    return abs


def get_date(root):
    date = xml_get_text(
        root.findall('./front/article-meta/pub-date/[@pub-type="epub"]'), "-"
    )

    if date == None:
        date = xml_get_text(
            root.findall('./front/article-meta/pub-date/[@date-type="pub"]'), 
            "-"
        )

    if date == None:
        date = xml_get_text(
            root.findall('./front/article-meta/pub-date/[@pub-type="ppub"]'), "-"
        )

    return date


def get_country(root):

    locations = [
        "./front/article-meta/aff/country",
        "./front/article-meta/contrib-group/aff/country",
        "./front/article-meta/contrib-group/contrib/aff/country",
        "./front/article-meta/aff",
        "./front/article-meta/contrib-group/aff",
        "./front/article-meta/contrib-group/contrib/aff",
    ]

    i = 0
    country = None

    while country == None:
        if i == len(locations):
            break

        country = xml_get_text(root.findall(locations[i]))
        # some queries return a whole adress, in which case we need the 
        # last word(s) of the string
        country = extract_country(country)

        i += 1

    return country


def extract_country(s: str) -> str:
    """
    Checks if the end of a string contains a valid country name
    This function is necessary bc some countries have names with 
    multiple words (e.g. United States).
    If there is no country at the end of the string, the function 
    returns None

    Args:
      s (str): string to be checked for countries
    
    Returns:
      str | None: if s contains a valid country, return the iso3166-
                  standardized country name, otherwise return None
    """

    if s in [None, ""]:
        return None

    s = s.split()
    for i in range(1, len(s) + 1):
        c = validate_country(" ".join(s[-i:]).strip(".,;"))
        if not c == None:
            return c

    return None


def validate_country(c: str) -> str:
    """
    Function to check if a string is a valid country name. 
    Strings are compared to the iso3166 standard. Some common country
    name versions are not recognized by iso, which is why they need to
    be manually transformed into a format that works with iso.

    Args:
      c (str): country to be checked
    
    Returns:
      str | None: if c is a country, return the iso3166-standardized 
                  country name, otherwise return None
    """

    # TO-DO: introduce case insensivity?

    if c == None:
        return None

    # country names that iso3166 doesn't recognize with their iso version
    country_alt_names = {
        "United States": "USA",
        "Republic of Korea": "KOR",
        "South Korea": "KOR",
        "United Kingdom": "GBR",
        "UK": "GBR",
        "Iran": "IRN",
        "Turkey": "TUR",
        "Vietnam": "VNM",
        "M&#x000e9;xico": "MEX",  # this doesn't work
        "Czech Republic": "CZE",
        "Russia": "RUS",
        "Bolivia": "BOL",
        "Tanzania": "TZA",
        "Venezuela": "VEN",
        "Brasil": "BRA",
    }

    if c in list(country_alt_names.keys()):
        c = country_alt_names[c]

    try:
        country = countries.get(c).apolitical_name
    except KeyError:
        country = None

    return country


"""
get_sections and get_sec_titles need to be combined!
why?
a) papers with len(sections) != len(section_titles) need to be discarded
   bc we can't make sure that the mapping is correct
   ALTERNATIVELY: already make a dict that combines each section with 
   the correct section title but then I would need to parse everything
   piece-wise
b) if the introduction does not have the section tag, it needs to be 
   separately parsed, and introduction needs to be added to the titles

"""
def get_sec_titles(root):

    titles = root.findall("./body/sec/title")
    title_names = [title.text for title in titles]
    if len(title_names) == 0:
        title_names = None

    return title_names


def get_sections_old(root):
    """So far, this only returns sections that have a section tag.
    However, many papers don't explicitly name their introduction and 
    instead start with the text in the body tag. This needs to be addressed!"""
    section_nodes = root.findall("./body/sec")

    secs = []
    for sec in section_nodes:
        secs.append(" ".join(sec.itertext()))

    return secs

def get_sections(root):
    """
    Parse sections and section titles from body of an xml file.
    Each title is supposed to be mapped to a section.
    If there are more sections than section titles, and the first 
    section is untitled, it is assumed to be the introduction. 
    Any further untitled sections are discarded.
    Otherwise, if there is text at the beginning of the body that is not 
    part of any section, it is assumed to be the introduction (this is 
    not ideal, because there is no way to check if it is indeed the 
    introduction and not some other section, such as abbreviations or
    key insights)

    Args:
      root: root of the files xtree
    
    Returns:
      dict: dictionary mapping each section to the corresponding section title
    """

    body = root.findall("./body")[0]
    secs = root.findall("./body/sec")

    # parse sections with titles
    sections_dict = {}
    for sec in secs:
        text = " ".join(sec.itertext())
        for child in sec:
            # only parse sections with title, to avoid mismatch
            if child.tag == "title":
                title = child.text
                sections_dict[title] = text
    
    # parse untitled sections
    if len(sections_dict) < len(secs):
        first_sec = " ".join(secs[0].itertext())
        first_sec_in_dict = list(sections_dict.values())[0]
        if not first_sec == first_sec_in_dict:
            sections_dict["Introduction"] = first_sec
    # parse text outside of sections
    else:
        intro = ""
        for child in body:
            if child.tag == "sec":
                break
            intro += " ".join(child.itertext())
        if len(intro) > 0:
            sections_dict["Introduction"] = intro

    return sections_dict
 

    





