"""Module to parse the pmc .xml files
Each file corresponds to one paper and contains meta information and the text itself"""

import xml.etree.ElementTree as et
import pandas as pd
import re
import os
from tqdm import tqdm
from iso3166 import countries

def xml_parse_single(xml_file: str) -> dict:
    """parses a single .xml file and stores the information in a dict"""

    paper_dict = {}

    xtree = et.parse(xml_file, parser=et.XMLParser(encoding="UTF-8"))
    xroot = xtree.getroot()

    paper_dict['article-type'] = xml_get_attr(xroot, 'article-type')
    paper_dict['language'] = xml_get_attr(xroot, '{http://www.w3.org/XML/1998/namespace}lang')
    paper_dict['journal'] = xml_get_text(xroot.findall('./front/journal-meta/journal-title-group/journal-title'))
    paper_dict['pmc-id'] = xml_get_text(xroot.findall('./front/article-meta/article-id/[@pub-id-type="pmc"]'))
    paper_dict['pmid'] = xml_get_text(xroot.findall('./front/article-meta/article-id/[@pub-id-type="pmid"]'))
    paper_dict['title'] = xml_get_text(xroot.findall('./front/article-meta/title-group/article-title'))
    paper_dict['country'] = get_country(xroot)
    paper_dict['date'] = get_date(xroot)
    paper_dict['abstract'] = get_abstr(xroot.findall('./front/article-meta/abstract'))
    paper_dict['section_titles'] = get_sec_titles(xroot)
    paper_dict['sections'] = get_sections(xroot)

    return paper_dict

def xml_parse_baseline_segment(data_path: str, json_path: str) -> pd.DataFrame:
    """parse all of the .xml files in a folder into one pandas dataframe, which is saved to a json file"""
    # set up dict to collect individiual papers
    keys = ['article-type', 'language', 'journal', 'pmc-id', 'pmid', 'title', 
            'country', 'date', 'abstract', 'section_titles', 'sections']
    baseline = {k:[] for k in keys}

    # exclude files that don't parse because of unbound prefixes
    exclude = ['PMC8494208.xml']

    for dirpath, _, filenames in os.walk(data_path):
        for file in tqdm(filenames):
            # check that only files named 'PMCxxxxxxxx.xml' are being processed
            r = re.compile('PMC\\d{6,8}.xml')
            if r.match(file) and not file in exclude:
                try:
                    paper_dict = xml_parse_single(os.path.join(dirpath, file))
                except et.ParseError as e:
                    print(f'parse error in file {file}: {e}')
                else:
                    # append new paper to baseline dict
                    {k:v.append(paper_dict[k]) for k,v in baseline.items()}

    baseline_df = pd.DataFrame.from_dict(baseline)
    baseline_df.to_json(json_path)

    return baseline_df

def xml_parse_baseline(data_path: str, json_path: str = '../data', replace: bool = False):
    """parse multiple folders of .xml files and save the extracted info in one .json file per folder
    data_path: location of .xml files
    json_path: location of .json output
    replace: if a .json file of same name already exists, should it be replaced?
    """

    dirname = os.path.basename(data_path)
    subdirs = next(os.walk(data_path))[1]

    for subdir in subdirs:
        subdir_path = os.path.join(data_path, subdir)
        json_file = os.path.join(json_path, dirname, subdir + '.json')
        
        print(f'parsing {subdir}')
        if replace or not os.path.exists(json_file):
            _ = xml_parse_baseline_segment(subdir_path, json_file)




### HELPER FUNCTIONS ###
def xml_get_attr(node, attr_name: str):
    if attr_name in node.attrib.keys():
        return node.attrib[attr_name]
    else:
        return None 
    

def xml_get_text(nodes, joinstr: str = ' '):
    if len(nodes) > 0:
        return joinstr.join(nodes[0].itertext()) #.text
    else: 
        return None
    

def get_abstr(node):
    abs = ""
    for a in node:
        if 'graphical' in a.attrib.values():
            continue 
        abs += xml_get_text(node)
    
    return abs

def get_date(root):
    date = xml_get_text(root.findall('./front/article-meta/pub-date/[@pub-type="epub"]'), '-')

    if date == None:
        date = xml_get_text(root.findall('./front/article-meta/pub-date/[@date-type="pub"]'), '-')
    
    if date == None:
        date = xml_get_text(root.findall('./front/article-meta/pub-date/[@pub-type="ppub"]'), '-')

    return date


def get_country(root):

    locations = ['./front/article-meta/aff/country',
                 './front/article-meta/contrib-group/aff/country',
                 './front/article-meta/contrib-group/contrib/aff/country',
                 './front/article-meta/aff',
                 './front/article-meta/contrib-group/aff',
                 './front/article-meta/contrib-group/contrib/aff',]
    
    i = 0
    country = None

    while country == None:
        if i == len(locations):
            break

        country = xml_get_text(root.findall(locations[i]))
        # some queries return a whole adress, in which case we need the last word(s) of the string 
        country = extract_country(country)
        
        i += 1
    
    return country

def extract_country(s: str) -> str:
    """checks if the end of a string contains a valid country name
    this function is necessary bc some countries have names with multiple words (e.g. united states)
    if there is no country at the end of the string, the function returns None"""

    if s in [None, '']:
        return None
    
    s = s.split()
    for i in range(1, len(s)+1):
        c = validate_country(' '.join(s[-i:]).strip('.,;'))
        if not c == None:
            return c
    
    return None


def validate_country(c: str) -> str:

    #TO-DO: introduce case insensivity?

    if c == None:
        return None
    
    # country names that iso3166 doesn't recognize with their iso version
    country_alt_names = {'United States': 'USA',
                         'Republic of Korea': 'KOR',
                         'South Korea': 'KOR',
                         'United Kingdom': 'GBR',
                         'UK': 'GBR',
                         'Iran': 'IRN',
                         'Turkey': 'TUR',
                         'Vietnam': 'VNM',
                         'M&#x000e9;xico': 'MEX', # this doesn't work
                         'Czech Republic': 'CZE',
                         'Russia': 'RUS',
                         'Bolivia': 'BOL',
                         'Tanzania': 'TZA',
                         'Venezuela': 'VEN',
                         'Brasil': 'BRA'}
    
    if c in list(country_alt_names.keys()):
        c = country_alt_names[c]
    
    try:
        country = countries.get(c).apolitical_name
    except KeyError:
        country = None
    
    return country

def get_sec_titles(root):

    titles = root.findall('./body/sec/title')
    title_names = [title.text for title in titles]
    if len(title_names) == 0:
        title_names = None

    return title_names

def get_sections(root):

    section_nodes = root.findall('./body/sec')

    secs = []
    for sec in section_nodes:
        secs.append(' '.join(sec.itertext()))

    return secs