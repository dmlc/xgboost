# -*- coding: utf-8 -*-
"""Helper utilty function for customization."""
import sys
import os
import docutils
import subprocess

READTHEDOCS_BUILD = (os.environ.get('READTHEDOCS', None) is not None)

if not os.path.exists('web-data'):
  subprocess.call('rm -rf web-data;' +
                  'git clone https://github.com/dmlc/web-data', shell = True)
else:
  subprocess.call('cd web-data; git pull', shell=True)

sys.stderr.write('READTHEDOCS=%s\n' % (READTHEDOCS_BUILD))
