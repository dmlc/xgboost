# -*- coding: utf-8 -*-
#
# documentation build configuration file, created by
# sphinx-quickstart on Thu Jul 23 19:40:08 2015.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.
from subprocess import call
from sh.contrib import git
import urllib.request
from urllib.error import HTTPError
from recommonmark.parser import CommonMarkParser
import sys
import re
import os
import subprocess
import guzzle_sphinx_theme

git_branch = os.getenv('SPHINX_GIT_BRANCH', default=None)
if git_branch is None:
    # If SPHINX_GIT_BRANCH environment variable is not given, run git
    # to determine branch name
    git_branch = [
        re.sub(r'origin/', '', x.lstrip(' ')) for x in str(
            git.branch('-r', '--contains', 'HEAD')).rstrip('\n').split('\n')
    ]
    git_branch = [x for x in git_branch if 'HEAD' not in x]
print(f'git_branch = {git_branch[0]}')
try:
    filename, _ = urllib.request.urlretrieve(
        f'https://s3-us-west-2.amazonaws.com/xgboost-docs/{git_branch[0]}.tar.bz2')
    call(f'if [ -d tmp ]; then rm -rf tmp; fi; mkdir -p tmp/jvm; cd tmp/jvm; tar xvf {filename}',
         shell=True)
except HTTPError:
    print('JVM doc not found. Skipping...')
try:
    filename, _ = urllib.request.urlretrieve(
        f'https://s3-us-west-2.amazonaws.com/xgboost-docs/doxygen/{git_branch[0]}.tar.bz2')
    call((f'mkdir -p tmp/dev; cd tmp/dev; tar xvf {filename}; ' +
          'mv doc_doxygen/html/* .; rm -rf doc_doxygen'), shell=True)
except HTTPError:
    print('C API doc not found. Skipping...')

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
libpath = os.path.join(curr_path, '../python-package/')
sys.path.insert(0, libpath)
sys.path.insert(0, curr_path)

# -- mock out modules
import mock                     # NOQA
MOCK_MODULES = ['scipy', 'scipy.sparse', 'sklearn', 'pandas']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# -- General configuration ------------------------------------------------

# General information about the project.
project = 'xgboost'
author = f'{project} developers'
copyright = '2020, {author}'
github_doc_root = 'https://github.com/dmlc/xgboost/tree/master/doc/'

os.environ['XGBOOST_BUILD_DOC'] = '1'
# Version information.
import xgboost                  # NOQA
version = xgboost.__version__
release = xgboost.__version__

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones
extensions = [
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'breathe'
]

graphviz_output_format = 'png'
plot_formats = [('svg', 300), ('png', 100), ('hires.png', 300)]
plot_html_show_source_link = False
plot_html_show_formats = False

# Breathe extension variables
breathe_projects = {"xgboost": "doxyxml/"}
breathe_default_project = "xgboost"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_parsers = {
  '.md': CommonMarkParser,
}

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md']

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

autoclass_content = 'both'

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']
html_extra_path = ['./tmp']

# The reST default role (used for this markup: `text`) to use for all
# documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme_path = guzzle_sphinx_theme.html_theme_path()
html_theme = 'guzzle_sphinx_theme'

# Register the theme as an extension to generate a sitemap.xml
extensions.append("guzzle_sphinx_theme")

# Guzzle theme options (see theme.conf for more information)
html_theme_options = {
    # Set the name of the project to appear in the sidebar
    "project_nav_name": "XGBoost"
}

html_sidebars = {
  '**': ['logo-text.html', 'globaltoc.html', 'searchbox.html']
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = project + 'doc'

# -- Options for LaTeX output ---------------------------------------------
latex_elements = {
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  (master_doc, f'{project}.tex', project,
   author, 'manual'),
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3.6', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
    'pandas': ('http://pandas-docs.github.io/pandas-docs-travis/', None),
    'sklearn': ('http://scikit-learn.org/stable', None)
}


# hook for doxygen
def run_doxygen(folder):
    """Run the doxygen make command in the designated folder."""
    try:
        retcode = subprocess.call("cd %s; make doxygen" % folder, shell=True)
        if retcode < 0:
            sys.stderr.write("doxygen terminated by signal %s" % (-retcode))
    except OSError as e:
        sys.stderr.write("doxygen execution failed: %s" % e)


def generate_doxygen_xml(app):
    """Run the doxygen make commands if we're on the ReadTheDocs server"""
    read_the_docs_build = os.environ.get('READTHEDOCS', None) == 'True'
    if read_the_docs_build:
        run_doxygen('..')


def setup(app):
    app.add_stylesheet('custom.css')
