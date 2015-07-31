# -*- coding: utf-8 -*-
"""Helper utilty function for customization."""
import sys
import os
import subprocess

if os.environ.get('READTHEDOCS', None) == 'True':
    subprocess.call('cd ..; rm -rf recommonmark recom;' +
                    'git clone https://github.com/tqchen/recommonmark;' +
                    'mv recommonmark/recommonmark recom', shell=True)

sys.path.insert(0, os.path.abspath('..'))
from recom import parser

class MarkdownParser(parser.CommonMarkParser):
    github_doc_root = None

    @staticmethod
    def remap_url(url):
        if MarkdownParser.github_doc_root is None or url is None:
            return url
        arr = url.split('#', 1)
        if arr[0].endswith('.md') and arr[0].find('://') == -1:
            arr[0] = arr[0][:-3] + '.html'
            return '#'.join(arr)
        else:
            return MarkdownParser.github_doc_root + url

    def reference(self, block):
        block.destination = remap_url(block.destination)
        return super(MarkdownParser, self).reference(block)

# inplace modify the function in recommonmark module to allow link remap
old_ref = parser.reference

def reference(block):
    block.destination = MarkdownParser.remap_url(block.destination)
    return old_ref(block)

parser.reference = reference
