# -*- coding: utf-8 -*-
"""Helper utilty function for customization."""
import sys
import os
import docutils
import subprocess

if os.environ.get('READTHEDOCS', None) == 'True':
    subprocess.call('cd ..; rm -rf recommonmark recom;' +
                    'git clone https://github.com/tqchen/recommonmark;' +
                    'mv recommonmark/recommonmark recom', shell=True)

sys.path.insert(0, os.path.abspath('..'))
from recom import parser, transform

MarkdownParser = parser.CommonMarkParser
AutoStructify = transform.AutoStructify
