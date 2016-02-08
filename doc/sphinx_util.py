# -*- coding: utf-8 -*-
"""Helper utilty function for customization."""
import sys
import os
import docutils
import subprocess

READTHEDOCS_BUILD = (os.environ.get('READTHEDOCS', None) is not None)

if not os.path.exists('../recommonmark'):
    subprocess.call('cd ..; rm -rf recommonmark;' +
                    'git clone https://github.com/tqchen/recommonmark', shell = True)
else:
    subprocess.call('cd ../recommonmark/; git pull', shell=True)

if not os.path.exists('web-data'):
    subprocess.call('rm -rf web-data;' +
                    'git clone https://github.com/dmlc/web-data', shell = True)
else:
    subprocess.call('cd web-data; git pull', shell=True)


sys.path.insert(0, os.path.abspath('../recommonmark/'))
sys.stderr.write('READTHEDOCS=%s\n' % (READTHEDOCS_BUILD))


from recommonmark import parser, transform

MarkdownParser = parser.CommonMarkParser
AutoStructify = transform.AutoStructify
