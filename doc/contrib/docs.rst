##########################
Documentation and Examples
##########################

**Contents**

.. contents::
  :backlinks: none
  :local:

*************
Documentation
*************
* Python and C documentation is built using `Sphinx <http://www.sphinx-doc.org/en/master/>`_.
* Each document is written in `reStructuredText <http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.
* The documentation is the ``doc/`` directory.
* You can build it locally using ``make html`` command.

  .. code-block:: bash

    make html

  Run ``make help`` to learn about the other commands.

The online document is hosted by `Read the Docs <https://readthedocs.org/>`__ where the imported project is managed by `Hyunsu Cho <https://github.com/hcho3>`__ and `Jiaming Yuan <https://github.com/trivialfis>`__.

=========================================
Build the Python Docs using pip and Conda
=========================================

#. Create a conda environment.

   .. code-block:: bash

     conda create -n xgboost-docs --yes python=3.10

   .. note:: Python 3.10 is required by `xgboost_ray <https://github.com/ray-project/xgboost_ray>`__ package.

#. Activate the environment

   .. code-block:: bash

     conda activate xgboost-docs

#. Install required packages (in the current environment) using ``pip`` command.

   .. code-block:: bash

     pip install -r requirements.txt

   .. note::
      It is currently not possible to install the required packages using ``conda``
      due to ``xgboost_ray`` being unavailable in conda channels.

      .. code-block:: bash

        conda install --file requirements.txt --yes -c conda-forge


#. (optional) Install `graphviz <https://www.graphviz.org/>`__

   .. code-block:: bash

     conda install graphviz --yes

#. Eventually, build the docs.

   .. code-block:: bash

     make html

  You should see the following messages in the console:

  .. code-block:: console

    $ make html
    sphinx-build -b html -d _build/doctrees   . _build/html
    Running Sphinx v6.2.1
    ...
    The HTML pages are in _build/html.

    Build finished. The HTML pages are in _build/html.

*************
Read The Docs
*************

`Read the Docs <https://readthedocs.org/>`__ (RTD for short) is an online document hosting
service and hosts the `XGBoost document site
<https://xgboost.readthedocs.io/en/stable/>`__. The document builder used by RTD is
relatively lightweight. However some of the packages like the R binding require a compiled
XGBoost along with all the optional dependencies to render the document. As a result, both
jvm-based packages and the R package's document is built with an independent CI pipeline
and fetched during online document build.

The sphinx configuration file ``xgboost/doc/conf.py`` acts as the fetcher. During build,
the fetched artifacts are stored in ``xgboost/doc/tmp/jvm_docs`` and
``xgboost/doc/tmp/r_docs`` respectively. For the R package, there's a dummy index file in
``xgboost/doc/R-package/r_docs`` . Jvm doc is similar. As for the C doc, it's generated
using doxygen and processed by breathe during build as it's relatively cheap. The
generated xml files are stored in ``xgboost/doc/tmp/dev`` .

The ``xgboost/doc/tmp`` is part of the ``html_extra_path`` sphinx configuration specified
in the ``conf.py`` file, which informs sphinx to copy the extracted html files to the
build directory. Following is a list of environment variables used by the fetchers in
``conf.py``:

 - ``READTHEDOCS``: Read the docs flag. Build the full documentation site including R, JVM and
   C doc when set to ``True`` (case sensitive).
 - ``XGBOOST_R_DOCS``: Local path for pre-built R document, used for development. If it
   points to a file that doesn't exist, the configuration script will download the
   packaged document to that path for future reuse.
 - ``XGBOOST_JVM_DOCS``: Local path for pre-built JVM document, used for
   development. Similar to the R docs environment variable when it points to a non-existent
   file.

As of writing, RTD doesn't provide any facility to be embedded as a GitHub action but we
need a way to specify the dependency between the CI pipelines and the document build in
order to fetch the correct artifact. The workaround is to use an extra GA step to notify
RTD using its `REST API <https://docs.readthedocs.com/platform/stable/api/v3.html>`__.

********
Examples
********
* Use cases and examples are in `demo <https://github.com/dmlc/xgboost/tree/master/demo>`_ directory.
* We are super excited to hear about your story. If you have blog posts,
  tutorials, or code solutions using XGBoost, please tell us, and we will add
  a link in the example pages.
