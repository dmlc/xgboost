# pylint: disable=invalid-name, exec-used
"""Convert README.md to README.rst for PyPI"""
from pypandoc import convert
read_md = convert('python-package/README.md', 'rst')
with open('python-package/README.rst', 'w') as rst_file:
    rst_file.write(read_md)
