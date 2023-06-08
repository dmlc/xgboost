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

The online document is hosted by `Read the Docs <https://readthedocs.org/>`__ where the imported project is managed by `Hyunsu Cho <https://github.com/hcho3>`__ and `Jiaming Yuan <https://github.com/trivialfis>`__.

===============================
Build Docs on macOS using Conda
===============================

#. Create a conda environment.

   .. code-block:: bash

     conda create -n xgboost-docs --yes python=3.10

   .. note:: Python 3.10 is required by `xgboost_ray <https://github.com/ray-project/xgboost_ray>`__ package.

#. Activate the environment

   .. code-block:: bash

     conda activate xgboost-docs

#. Install required packages (in the current environment) using ``pip`` command.
   For some reason, it is currently not possible to install the required packages using ``conda``.

   .. code-block:: bash

     pip install -r requirements.txt

#. (optional) Install graphviz

   .. code-block:: bash

     brew install graphviz

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

********
Examples
********
* Use cases and examples are in `demo <https://github.com/dmlc/xgboost/tree/master/demo>`_ directory.
* We are super excited to hear about your story. If you have blog posts,
  tutorials, or code solutions using XGBoost, please tell us, and we will add
  a link in the example pages.
