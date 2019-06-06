#!C:\Users\user\Documents\qpr\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'nbformat==4.4.0','console_scripts','jupyter-trust'
__requires__ = 'nbformat==4.4.0'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('nbformat==4.4.0', 'console_scripts', 'jupyter-trust')()
    )
