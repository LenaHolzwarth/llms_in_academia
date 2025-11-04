"""Module to parse the pmc .xml files
Each file corresponds to one paper and contains meta information and the text itself"""

import xml.etree.ElementTree as et
import pandas as pd
import re
import os
from tqdm import tqdm

def xml_parse_single(xml_file: str) -> dict:
    """parses a single .xml file and stores the information in a dict"""

    paper_dict = {}

    xtree = et.parse(xml_file, parser=et.XMLParser(encoding="UTF-8"))
    xroot = xtree.getroot()

    paper_dict['article-type'] = xml_get_attr(xroot, 'article-type')
    paper_dict['language'] = xml_get_attr(xroot, '{http://www.w3.org/XML/1998/namespace}lang')
    paper_dict['journal'] = xml_get_text(xroot.findall('./front/journal-meta/journal-title-group/journal-title'))
    paper_dict['pmc-id'] = xml_get_text(xroot.findall('./front/article-meta/article-id/[@pub-id-type="pmc"]'))
    paper_dict['title'] = xml_get_text(xroot.findall('./front/article-meta/title-group/article-title'))
    paper_dict['country'] = get_country(xroot)
    paper_dict['date'] = get_date(xroot)
    paper_dict['abstract'] = get_abstr(xroot.findall('./front/article-meta/abstract'))

    return paper_dict

def xml_parse_baseline(data_path: str, json_path: str) -> pd.DataFrame:

    # set up dict to collect individiual papers
    keys = ['article-type', 'language', 'journal', 'pmc-id', 'title', 'country', 'date', 'abstract']
    baseline = {k:[] for k in keys}

    for dirpath, _, filenames in os.walk(data_path):
        for file in tqdm(filenames):
            # check that only files named 'PMCxxxxxxxx.xml' are being processed
            r = re.compile('PMC\d{8}.xml')
            if r.match(file):
                paper_dict = xml_parse_single(os.path.join(dirpath, file))
                # append new paper to baseline dict
                {k:v.append(paper_dict[k]) for k,v in baseline.items()}

    baseline_df = pd.DataFrame.from_dict(baseline)
    baseline_df.to_json(json_path)

    return baseline_df


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

def get_country_old(root):
    
    # try first location
    country = xml_get_text(root.findall('./front/article-meta/aff/country'))

    if country == None:
        #try second location
        country = xml_get_text(root.findall('./front/article-meta/aff/[@id="I1"]'))
        if not country == None:
            country = country.split()
            if len(country) > 0:
                country = country[-1]
            else:
                country = None
    
    if country == None:
        #try third location
        country = xml_get_text(root.findall('./front/article-meta/contrib-group/contrib/[@corresp="yes"]/aff/country'))
      
    # prevent two names for same country
    if country == 'United States':
        country = 'USA'
    
    return country

def get_country(root):

    locations = ['./front/article-meta/aff/country',
                 './front/article-meta/aff',
                 './front/article-meta/contrib-group/contrib/aff/country',
                 './front/article-meta/contrib-group/aff']
    
    i = 0
    country = None

    while country == None:
        if i == len(locations):
            break

        country = xml_get_text(root.findall(locations[i]))
        i += 1
    
    # some queries return a whole adress, in which case we need the last word of the string 
    if not country == None:
        country = country.split()
        if len(country) > 0:        # check that the string is non-empty
            country = country[-1]
        else:
            country = None          # replace empty string with None
        
    
    return country