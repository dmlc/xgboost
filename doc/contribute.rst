#####################
Contribute to XGBoost
#####################
XGBoost has been developed and used by a group of active community members.
Everyone is more than welcome to contribute. It is a way to make the project better and more accessible to more users.

- Please add your name to `CONTRIBUTORS.md <https://github.com/dmlc/xgboost/blob/master/CONTRIBUTORS.md>`_ after your patch has been merged.
- Please also update `NEWS.md <https://github.com/dmlc/xgboost/blob/master/NEWS.md>`_ to add note on your changes to the API or XGBoost documentation.

**Guidelines**

* `Submit Pull Request`_
* `Git Workflow Howtos`_

  - `How to resolve conflict with master`_
  - `How to combine multiple commits into one`_
  - `What is the consequence of force push`_

* `Documents`_
* `Testcases`_
* `Sanitizers`_
* `Examples`_
* `Core Library`_
* `Python Package`_
* `R Package`_

*******************
Submit Pull Request
*******************

* Before submit, please rebase your code on the most recent version of master, you can do it by

  .. code-block:: bash

    git remote add upstream https://github.com/dmlc/xgboost
    git fetch upstream
    git rebase upstream/master

* If you have multiple small commits,
  it might be good to merge them together(use git rebase then squash) into more meaningful groups.
* Send the pull request!

  - Fix the problems reported by automatic checks
  - If you are contributing a new module, consider add a testcase in `tests <https://github.com/dmlc/xgboost/tree/master/tests>`_.

*******************
Git Workflow Howtos
*******************

How to resolve conflict with master
===================================
- First rebase to most recent master

  .. code-block:: bash

    # The first two steps can be skipped after you do it once.
    git remote add upstream https://github.com/dmlc/xgboost
    git fetch upstream
    git rebase upstream/master

- The git may show some conflicts it cannot merge, say ``conflicted.py``.

  - Manually modify the file to resolve the conflict.
  - After you resolved the conflict, mark it as resolved by

    .. code-block:: bash

      git add conflicted.py

- Then you can continue rebase by

  .. code-block:: bash

    git rebase --continue

- Finally push to your fork, you may need to force push here.

  .. code-block:: bash

    git push --force

How to combine multiple commits into one
========================================
Sometimes we want to combine multiple commits, especially when later commits are only fixes to previous ones,
to create a PR with set of meaningful commits. You can do it by following steps.

- Before doing so, configure the default editor of git if you haven't done so before.

  .. code-block:: bash

    git config core.editor the-editor-you-like

- Assume we want to merge last 3 commits, type the following commands

  .. code-block:: bash

    git rebase -i HEAD~3

- It will pop up an text editor. Set the first commit as ``pick``, and change later ones to ``squash``.
- After you saved the file, it will pop up another text editor to ask you modify the combined commit message.
- Push the changes to your fork, you need to force push.

  .. code-block:: bash

    git push --force

What is the consequence of force push
=====================================
The previous two tips requires force push, this is because we altered the path of the commits.
It is fine to force push to your own fork, as long as the commits changed are only yours.

*********
Documents
*********
* Documentation is built using sphinx.
* Each document is written in `reStructuredText <http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_.
* You can build document locally to see the effect.

*********
Testcases
*********
* All the testcases are in `tests <https://github.com/dmlc/xgboost/tree/master/tests>`_.
* We use python nose for python test cases.

**********
Sanitizers
**********

By default, sanitizers are bundled in GCC and Clang/LLVM. One can enable
sanitizers with GCC >= 4.8 or LLVM >= 3.1, But some distributions might package
sanitizers separately.  Here is a list of supported sanitizers with
corresponding library names:

- Address sanitizer: libasan
- Leak sanitizer:    liblsan
- Thread sanitizer:  libtsan

Memory sanitizer is exclusive to LLVM, hence not supported in XGBoost.

How to build XGBoost with sanitizers
====================================
One can build XGBoost with sanitizer support by specifying -DUSE_SANITIZER=ON.
By default, address sanitizer and leak sanitizer are used when you turn the
USE_SANITIZER flag on.  You can always change the default by providing a
semicolon separated list of sanitizers to ENABLED_SANITIZERS.  Note that thread
sanitizer is not compatible with the other two sanitizers.

  .. code-block:: bash

    cmake -DUSE_SANITIZER=ON -DENABLED_SANITIZERS="address;leak" /path/to/xgboost

By default, CMake will search regular system paths for sanitizers, you can also
supply a specified SANITIZER_PATH.

  .. code-block:: bash

    cmake -DUSE_SANITIZER=ON -DENABLED_SANITIZERS="address;leak" \
    -DSANITIZER_PATH=/path/to/sanitizers /path/to/xgboost

How to use sanitizers with CUDA support
=======================================
Runing XGBoost on CUDA with address sanitizer (asan) will raise memory error.
To use asan with CUDA correctly, you need to configure asan via ASAN_OPTIONS
environment variable:

  .. code-block:: bash

    ASAN_OPTIONS=protect_shadow_gap=0 ../testxgboost

For details, please consult `official documentation <https://github.com/google/sanitizers/wiki>`_ for sanitizers.


********
Examples
********
* Usecases and examples will be in `demo <https://github.com/dmlc/xgboost/tree/master/demo>`_.
* We are super excited to hear about your story, if you have blogposts,
  tutorials code solutions using XGBoost, please tell us and we will add
  a link in the example pages.

************
Core Library
************
- Follow `Google style for C++ <https://google.github.io/styleguide/cppguide.html>`_.
- Use C++11 features such as smart pointers, braced initializers, lambda functions, and ``std::thread``.
- We use Doxygen to document all the interface code.
- You can reproduce the linter checks by running ``make lint``

**************
Python Package
**************
- Always add docstring to the new functions in numpydoc format.
- You can reproduce the linter checks by typing ``make lint``

*********
R Package
*********

Code Style
==========
- We follow Google's C++ Style guide for C++ code.

  - This is mainly to be consistent with the rest of the project.
  - Another reason is we will be able to check style automatically with a linter.

- You can check the style of the code by typing the following command at root folder.

  .. code-block:: bash

    make rcpplint

- When needed, you can disable the linter warning of certain line with ```// NOLINT(*)``` comments.
- We use `roxygen <https://cran.r-project.org/web/packages/roxygen2/vignettes/roxygen2.html>`_ for documenting the R package.

Rmarkdown Vignettes
===================
Rmarkdown vignettes are placed in `R-package/vignettes <https://github.com/dmlc/xgboost/tree/master/R-package/vignettes>`_.
These Rmarkdown files are not compiled. We host the compiled version on `doc/R-package <https://github.com/dmlc/xgboost/tree/master/doc/R-package>`_.

The following steps are followed to add a new Rmarkdown vignettes:

- Add the original rmarkdown to ``R-package/vignettes``.
- Modify ``doc/R-package/Makefile`` to add the markdown files to be build.
- Clone the `dmlc/web-data <https://github.com/dmlc/web-data>`_ repo to folder ``doc``.
- Now type the following command on ``doc/R-package``:

  .. code-block:: bash

    make the-markdown-to-make.md

- This will generate the markdown, as well as the figures in ``doc/web-data/xgboost/knitr``.
- Modify the ``doc/R-package/index.md`` to point to the generated markdown.
- Add the generated figure to the ``dmlc/web-data`` repo.

  - If you already cloned the repo to doc, this means ``git add``

- Create PR for both the markdown  and ``dmlc/web-data``.
- You can also build the document locally by typing the following command at the ``doc`` directory:

  .. code-block:: bash

    make html

The reason we do this is to avoid exploded repo size due to generated images.

R package versioning
====================
Since version 0.6.4.3, we have adopted a versioning system that uses x.y.z (or ``core_major.core_minor.cran_release``)
format for CRAN releases and an x.y.z.p (or ``core_major.core_minor.cran_release.patch``) format for development patch versions.
This approach is similar to the one described in Yihui Xie's
`blog post on R Package Versioning <https://yihui.name/en/2013/06/r-package-versioning/>`_,
except we need an additional field to accomodate the x.y core library version.

Each new CRAN release bumps up the 3rd field, while developments in-between CRAN releases
would be marked by an additional 4th field on the top of an existing CRAN release version.
Some additional consideration is needed when the core library version changes.
E.g., after the core changes from 0.6 to 0.7, the R package development version would become 0.7.0.1, working towards
a 0.7.1 CRAN release. The 0.7.0 would not be released to CRAN, unless it would require almost no additional development.

Registering native routines in R
================================
According to `R extension manual <https://cran.r-project.org/doc/manuals/r-release/R-exts.html#Registering-native-routines>`_,
it is good practice to register native routines and to disable symbol search. When any changes or additions are made to the
C++ interface of the R package, please make corresponding changes in ``src/init.c`` as well.
