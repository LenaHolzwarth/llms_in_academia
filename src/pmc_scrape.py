""""Module to download the current baseline of pubmed central open access papers"""

from ftplib import FTP
import re
import os
from tqdm import tqdm 

PMC_URL = 'ftp.ncbi.nlm.nih.gov'
PMC_OA_PATH = '/pub/pmc/oa_bulk/'
OA_SECTIONS = ['oa_comm', 'oa_noncomm']

def scrape_baseline(output_path: str, 
                    get_baseline: bool = True, 
                    get_filelist: bool = False):
    """download current baseline of pmc oa papers"""
    for sec in OA_SECTIONS:
        # get list of all files for current section
        ftp = FTP(PMC_URL)
        ftp.login()
        ftp.cwd(PMC_OA_PATH + sec + '/xml')
    
        filenames = ftp.nlst()
        ftp.quit() 

        # determine which files to download
        if get_baseline:
            if get_filelist:
                # remove increment packages
                filenames = [x for x in filenames if 'baseline' in x and '.txt' not in x]
            else:
                # remove file lists and increment packages
                filenames = [x for x in filenames if 'tar.gz' in x and 'baseline' in x]
        else:
            if get_filelist:
                filenames = [x for x in filenames if '.csv' in x and 'baseline' in x]
            else:
                raise ValueError(f"get_filelist and get_baseline can't both be False")

        # get date of current baseline and create folder in output path
        baseline_date = re.findall(r'\d{4}\-\d{2}\-\d{2}', filenames[0])[0]
        output_dir = os.path.join(output_path, 'baseline_' + baseline_date)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        for package_name in filenames:
            
            output_file = os.path.join(output_dir, package_name)
            if os.path.exists(output_file):
                print(f'package {package_name} already downloaded')
            else:
                print(f'downloading package {package_name}')
                scrape_package_tqdm(package_name, sec, output_file)
                print('download complete')


        


def scrape_package(package_name: str, oa_subsection: str, output_path: str):
    """download single package of pmc oa papers"""
    if oa_subsection not in OA_SECTIONS:
        raise ValueError(f"subsection {oa_subsection} doesn't exist")

    
    # open ftp connection
    ftp = FTP(PMC_URL)
    ftp.login()
    ftp.cwd(PMC_OA_PATH + oa_subsection + '/xml')

    # download package
    with open(output_path, 'wb') as fp:
        ftp.retrbinary('RETR %s' % package_name, fp.write)
        fp.close

    ftp.quit()

def scrape_package_tqdm(package_name: str, oa_subsection: str, output_path: str):
    """download single package of pmc oa papers"""
    if oa_subsection not in OA_SECTIONS:
        raise ValueError(f"subsection {oa_subsection} doesn't exist")

    remote_path = os.path.join(PMC_OA_PATH + oa_subsection + '/xml', package_name)

    with FTP(PMC_URL, user="anonymous") as ftp:
            file_size = ftp.size(remote_path)
            with open(output_path, "wb") as local_file:
                with tqdm(
                    total=file_size, unit="B", unit_scale=True, desc="Downloading"
                ) as progress_bar:

                    def write_with_tqdm(chunk):
                        local_file.write(chunk)
                        progress_bar.update(len(chunk))

                    ftp.retrbinary(
                        f"RETR {remote_path}", write_with_tqdm, blocksize=32768
                    )
