#!C:\Users\user\Documents\qpr\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'hepqpr-qallse==0.1.0','console_scripts','create_dataset'
__requires__ = 'hepqpr-qallse==0.1.0'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('hepqpr-qallse==0.1.0', 'console_scripts', 'create_dataset')()
    )
