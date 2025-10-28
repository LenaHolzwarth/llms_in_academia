""""Module to download the current baseline of pubmed central open access papers"""

from ftplib import FTP

PMC_URL = 'ftp.ncbi.nlm.nih.gov'
PMC_OA_PATH = '/pub/pmc/oa_bulk/'
OA_SECTIONS = ['oa_comm', 'oa_noncomm']

def scrape_baseline():
    """download current baseline of pmc oa papers"""
    for sec in OA_SECTIONS:
        # get list of all baseline packages for current section
        ftp = FTP(PMC_URL)
        ftp.login()
        ftp.cwd(PMC_OA_PATH + sec + '/xml')
    
        filenames = ftp.nlst()

        ftp.quit()


def scrape_package(package_name: str, oa_subsection: str, output_path: str):
    """download single package of pmc oa papers"""
    #TODO: make this an exception
    if oa_subsection not in OA_SECTIONS:
        return f"subsection {oa_subsection} doesn't exist"
    
    # open ftp connection
    ftp = FTP(PMC_URL)
    ftp.login()
    ftp.cwd(PMC_OA_PATH + oa_subsection + '/xml')

    # download package
    with open(output_path + package_name, 'wb') as fp:
        ftp.retrbinary('RETR %s' % package_name, fp.write)
        fp.close

    ftp.quit()