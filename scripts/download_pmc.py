"""script for downloading pmc open access papers"""

import sys
from src.pmc_scrape import scrape_baseline

path_local = '/Users/lena/documents/GitHub/llms_in_academia/data/'
path_remote = '/Users/lena/documents/cinmount-data/'

if len(sys.argv) > 1:
    use_local = sys.argv[1]
else:
    use_local = False

if use_local:
    output_path = path_local
else:
    output_path = path_remote

scrape_baseline(output_path, get_baseline=True, get_filelist=False)