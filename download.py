import urllib
import tarfile
import phoebe
import os
import sys
import subprocess

    

if __name__ == "__main__":
    files = None
    types = 'all'
    if sys.argv[1:]:
        types = sys.argv[1]
        if sys.argv[2:]:
            files = sys.argv[2]
    
    if types == 'atm' or types == 'all':
        phoebe.atmospheres.limbdark.download_atm(files)
