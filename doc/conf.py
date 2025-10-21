"""Sphinx configuration.

See `doc/contrib/docs.rst <https://xgboost.readthedocs.io/en/stable/contrib/docs.html>`__
for more info.
"""

import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import warnings
from urllib.error import HTTPError

CURR_PATH = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
PROJECT_ROOT = os.path.normpath(os.path.join(CURR_PATH, os.path.pardir))
TMP_DIR = os.path.join(CURR_PATH, "tmp")
DOX_DIR = "doxygen"

# Directly load the source module.
sys.path.append(os.path.join(PROJECT_ROOT, "python-package"))
# Tell xgboost to not load the libxgboost.so
os.environ["XGBOOST_BUILD_DOC"] = "1"

# Version information.
import xgboost  # NOQA

version = xgboost.__version__
release = xgboost.__version__


# Document is uploaded to here by the CI builder.
S3_BUCKET = "https://xgboost-docs.s3.us-west-2.amazonaws.com"


def run_doxygen() -> None:
    """Run the doxygen make command in the designated folder."""
    curdir = os.path.normpath(os.path.abspath(os.path.curdir))
    if os.path.exists(TMP_DIR):
        print(f"Delete directory {TMP_DIR}")
        shutil.rmtree(TMP_DIR)
    else:
        print(f"Create directory {TMP_DIR}")
        os.mkdir(TMP_DIR)
    try:
        os.chdir(PROJECT_ROOT)
        if not os.path.exists(DOX_DIR):
            os.mkdir(DOX_DIR)
        os.chdir(os.path.join(PROJECT_ROOT, DOX_DIR))
        print(
            "Build doxygen at {}".format(
                os.path.join(PROJECT_ROOT, DOX_DIR, "doc_doxygen")
            )
        )
        subprocess.check_call(["cmake", "..", "-DBUILD_C_DOC=ON", "-GNinja"])
        subprocess.check_call(["ninja", "doc_doxygen"])

        src = os.path.join(PROJECT_ROOT, DOX_DIR, "doc_doxygen", "html")
        dest = os.path.join(TMP_DIR, "dev")
        print(f"Copy directory {src} -> {dest}")
        shutil.copytree(src, dest)
    except OSError as e:
        sys.stderr.write("doxygen execution failed: %s" % e)
    finally:
        os.chdir(curdir)


def get_branch() -> str:
    """Guess the git branch."""
    branch = os.getenv("READTHEDOCS_VERSION_NAME", default=None)
    print(f"READTHEDOCS_VERSION_NAME = {branch}")

    def is_id():
        try:
            return str(int(branch)) == branch
        except ValueError:
            return False

    if not branch:  # Not in RTD
        branch = "master"  # use the master branch as the default.
    elif branch == "latest":
        branch = "master"
    elif branch.startswith("release_"):
        pass  # release branch, like: release_2.1.0
    elif branch == "stable":
        # Avoid patch release branch.
        v = xgboost.__version__.split(".")
        branch = f"release_{v[0]}.{v[1]}.0"
    elif is_id():
        # Likely PR branch
        branch = f"PR-{branch}"
    else:  # other dmlc branches.
        pass
    print(f"branch = {branch}")
    return branch


def get_sha(branch: str) -> str | None:
    sha = os.getenv("READTHEDOCS_GIT_COMMIT_HASH", default=None)
    if sha is not None:
        return sha

    if branch == "master":
        res = subprocess.run(["git", "rev-parse", "master"], stdout=subprocess.PIPE)
    else:
        res = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
    if res.returncode != 0:
        return None
    return res.stdout.decode("utf-8").strip()


def download_jvm_docs() -> None:
    """Fetch docs for the JVM packages"""
    print("Download JVM documents from S3.")
    branch = get_branch()
    commit = get_sha(branch)
    if commit is None:
        print("Couldn't find commit to build jvm docs.")
        return

    def try_fetch_jvm_doc(branch: str) -> bool:
        """
        Attempt to fetch JVM docs for a given branch.
        Returns True if successful
        """
        try:
            local_jvm_docs = os.environ.get("XGBOOST_JVM_DOCS", None)
            url = f"{S3_BUCKET}/{branch}/{commit}/{branch}.tar.bz2"
            if local_jvm_docs is not None:
                local_jvm_docs = os.path.expanduser(local_jvm_docs)

            if local_jvm_docs is not None and os.path.exists(local_jvm_docs):
                # Reuse an existing tarball.
                filename = local_jvm_docs
            elif local_jvm_docs is not None:
                # Download to local_jvm_docs for future reuse.
                filename, _ = urllib.request.urlretrieve(url, filename=local_jvm_docs)
                print(f"Finished: {url} -> {filename}")
            else:
                filename, _ = urllib.request.urlretrieve(url)
                print(f"Finished: {url} -> {filename}")
            if not os.path.exists(TMP_DIR):
                os.mkdir(TMP_DIR)
            jvm_doc_dir = os.path.join(TMP_DIR, "jvm_docs")
            if os.path.exists(jvm_doc_dir):
                shutil.rmtree(jvm_doc_dir)
            os.mkdir(jvm_doc_dir)

            with tarfile.open(filename, "r:bz2") as t:
                t.extractall(jvm_doc_dir)
            return True
        except HTTPError:
            print(f"JVM doc not found at {url}. Skipping...")
            return False

    if not try_fetch_jvm_doc(branch):
        print("Falling back to the master branch.")
        try_fetch_jvm_doc("master")


def download_r_docs() -> None:
    """Fetch R document from s3."""
    branch = get_branch()
    commit = get_sha(branch)
    print("Download R documents from S3.")
    if commit is None:
        print("Couldn't find commit to build R docs.")
        return

    def try_fetch_r_doc(branch: str) -> bool:
        try:
            local_r_docs = os.environ.get("XGBOOST_R_DOCS", None)
            url = f"{S3_BUCKET}/{branch}/{commit}/r-docs-{branch}.tar.bz2"
            if local_r_docs is not None:
                local_r_docs = os.path.expanduser(local_r_docs)

            if local_r_docs is not None and os.path.exists(local_r_docs):
                # Reuse an existing tarball.
                filename = local_r_docs
            elif local_r_docs is not None:
                filename, _ = urllib.request.urlretrieve(url, filename=local_r_docs)
                print(f"Finished: {url} -> {filename}")
            else:
                filename, _ = urllib.request.urlretrieve(url)
                print(f"Finished: {url} -> {filename}")

            if not os.path.exists(TMP_DIR):
                os.mkdir(TMP_DIR)
            r_doc_dir = os.path.join(TMP_DIR, "r_docs")
            if os.path.exists(r_doc_dir):
                shutil.rmtree(r_doc_dir)
            os.mkdir(r_doc_dir)

            with tarfile.open(filename, "r:bz2") as t:
                t.extractall(r_doc_dir)

            for root, subdir, files in os.walk(
                os.path.join(r_doc_dir, "doc", "R-package")
            ):
                for f in files:
                    assert f.endswith(".md")
                    src = os.path.join(root, f)
                    dst = os.path.join(PROJECT_ROOT, "doc", "R-package", f)
                    shutil.move(src, dst)
            return True
        except HTTPError:
            print(f"R doc not found at {url}. Falling back to the master branch.")
            return False

    if not try_fetch_r_doc(branch):
        try_fetch_r_doc("master")


def is_readthedocs_build():
    if os.environ.get("READTHEDOCS", None) == "True":
        return True
    warnings.warn(
        "Skipping Doxygen build... You won't have documentation for C/C++ functions. "
        "Set environment variable READTHEDOCS=True if you want to build Doxygen. "
        "(If you do opt in, make sure to install Doxygen, Graphviz, CMake, and C++ compiler "
        "on your system.)"
    )
    return False


if is_readthedocs_build():
    run_doxygen()
    download_jvm_docs()
    download_r_docs()


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
libpath = os.path.join(PROJECT_ROOT, "python-package/")
sys.path.insert(0, libpath)
sys.path.insert(0, CURR_PATH)

# -- General configuration ------------------------------------------------

# General information about the project.
project = "xgboost"
author = "%s developers" % project
copyright = "2025, %s" % author
github_doc_root = "https://github.com/dmlc/xgboost/tree/master/doc/"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones
extensions = [
    "matplotlib.sphinxext.plot_directive",
    "sphinxcontrib.jquery",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "sphinx_issues",
    "sphinx_tabs.tabs",
    "breathe",
    "myst_parser",
]

sphinx_gallery_conf = {
    # path to your example scripts
    "examples_dirs": [
        "../demo/guide-python",
        "../demo/dask",
        "../demo/aft_survival",
        "../demo/rmm_plugin",
    ],
    # path to where to save gallery generated output
    "gallery_dirs": [
        "python/examples",
        "python/dask-examples",
        "python/survival-examples",
        "python/rmm-examples",
    ],
    "matplotlib_animations": True,
}

# Sphinx-issues configuration
# Path to GitHub repo {group}/{project}  (note that `group` is the GitHub user or organization)
issues_github_path = "dmlc/xgboost"

autodoc_typehints = "description"

graphviz_output_format = "png"
plot_formats = [("svg", 300), ("png", 100), ("hires.png", 300)]
plot_html_show_source_link = False
plot_html_show_formats = False

# Breathe extension variables
breathe_projects = {}
if is_readthedocs_build():
    breathe_projects = {
        "xgboost": os.path.join(PROJECT_ROOT, DOX_DIR, "doc_doxygen/xml")
    }
breathe_default_project = "xgboost"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst", ".md"]

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

autoclass_content = "both"

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]
html_extra_path = []
if is_readthedocs_build():
    html_extra_path = [TMP_DIR]

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
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"
html_theme_options = {"logo_only": True}


html_logo = "https://xgboost.ai/images/logo/xgboost-logo.png"

html_css_files = ["css/custom.css"]

html_sidebars = {"**": ["logo-text.html", "globaltoc.html", "searchbox.html"]}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Output file base name for HTML help builder.
htmlhelp_basename = project + "doc"

# -- Options for LaTeX output ---------------------------------------------
latex_elements = {}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "%s.tex" % project, project, author, "manual"),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
    "distributed": ("https://distributed.dask.org/en/stable/", None),
    "pyspark": ("https://spark.apache.org/docs/latest/api/python/", None),
    "rmm": ("https://docs.rapids.ai/api/rmm/nightly/", None),
}


def setup(app):
    app.add_css_file("custom.css")
