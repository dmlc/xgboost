# -*- coding: utf-8 -*-
"""Helper hacking utilty function for customization."""
import sys
import os
import subprocess

# TODO: make less hacky way than this one
if os.environ.get('READTHEDOCS', None) == 'True':
    subprocess.call('cd ..; rm -rf recommonmark;' +
                    'git clone https://github.com/tqchen/recommonmark;' +
                    'cp recommonmark/recommonmark/parser.py doc/parser', shell=True)

sys.path.insert(0, os.path.abspath('..'))
import parser

class MarkdownParser(parser.CommonMarkParser):
    github_doc_root = None
    doc_suffix = set(['md', 'rst'])

    @staticmethod
    def remap_url(url):
        if MarkdownParser.github_doc_root is None or url is None:
            return url
        if url.startswith('#'):
            return url
        arr = url.split('#', 1)
        ssuffix = arr[0].rsplit('.', 1)

        if len(ssuffix) == 2 and (ssuffix[-1] in MarkdownParser.doc_suffix
                                  and arr[0].find('://') == -1):
            arr[0] = ssuffix[0] + '.html'
            return '#'.join(arr)
        else:
            if arr[0].find('://') == -1:
                return MarkdownParser.github_doc_root + url
            else:
                return url

    def reference(self, block):
        block.destination = remap_url(block.destination)
        return super(MarkdownParser, self).reference(block)

# inplace modify the function in recommonmark module to allow link remap
old_ref = parser.reference

def reference(block):
    block.destination = MarkdownParser.remap_url(block.destination)
    return old_ref(block)

parser.reference = reference
