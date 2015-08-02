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

class MarkdownParser(docutils.parsers.Parser):
    github_doc_root = None


    def __init__(self):
        self.parser = parser.CommonMarkParser()

    def parse(self, inputstring, document):
        self.parser.parse(inputstring, document)
        transform.AutoStructify.url_resolver = [resolve_url]
        for trans in self.get_transforms():
            transform.AutoStructify(document).apply()

    def get_transforms(self):
        return [transform.AutoStructify]

def resolve_url(url):
    if MarkdownParser.github_doc_root is None or url is None:
        return url
    else:
        return MarkdownParser.github_doc_root + url
